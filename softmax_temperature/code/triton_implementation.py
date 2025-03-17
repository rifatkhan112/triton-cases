import torch
import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

######### Step 4 #########
@triton.jit 
def _softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,    # number of elements to skip when moving to next row
    n_rows, n_cols,                         # matrix dimensions
    BLOCK_SIZE: tl.constexpr,               # lowest power-of-2 greater than n_cols
    num_stages: tl.constexpr,
): 
    # The pid defines the row that this program starts with
    row_start = tl.program_id(0) 
    # then this gets the total number of parallel programs, which we'll use to know how large 
    #  of a step to make in our for loop once we finish the first row
    row_step = tl.num_programs(0) 
        # Each program processes rows strided by row_step 
        # (ex. if there are 4 programs, program 0 handles rows 0,4,8...)
    
    # whereas tl.arange() provides an array of values, tl.range() acts as an iterator
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # Rather than implement each iteration of the for loop sequentially, triton can use
        #  num_stages to work on different iterations of the for loop simultaneously. Of course
        #  Only do this when the iterations don't depend on each other
        
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
            # inyuiyively input_row_stride should be one as long as the input tensor is contiguous.
            #  but what if a non-contiguous view of a manipulated tensor were passed in? then
            #  input_row_stride matters

        # load the row into SRAM, using a mask since BLOCK_SIZE is > than n_cols if n_cols is not a power of 2
        col_offsets = tl.arange(0, BLOCK_SIZE) # we can fit each row in a single block
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=float('-inf')) 
            # We fill in masked-out indices with -inf since that's the value that won't influence softmax
        # Temperature parameter 
        tau = 100
        # subtract maximum for numerical stability
        row_minus_max = (row - tl.max(row, axis=0)) / tau
            # all the invalid -inf values remain -inf when we subtract the max
        # note that exponentiation in Triton is fast but approximate; later, we'll learn an even faster alternative
        numerator = tl.exp(row_minus_max)
            # all the -inf values get set to 0 since exp(-inf)=0
        denominator = tl.sum(numerator, axis=0)
            # All the invalid zero values do get summed, but they don't matter since they're 0
        softmax_output = numerator / denominator
            # All the invalid 0's are 0/sum and therefore remain 0

        # write output back to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)
            # using our mask, we only store back the valid n_cols values

######### Step 3 #########
"""
before we create the wrapper function that enqueues the kernel and its meta-parameters, we're going to
 fetch the specifications of our GPU to help later when defining our meta-parameters such that they're 
 especially well suited (fast) to the specific GPU we're using
"""
# fetching a dictionary full of the GPU's specifications
properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
# each Streaming Multi-processor (SM) is like a mini-processor that can run multiple programs
NUM_SM = properties["multiprocessor_count"] 
# registers are the fastest memory on the GPU
NUM_REGS = properties["max_num_regs"] 
    # each SM has a limited number of registers; 
    # Programs share these registers, so using too many per program limits parallelism
# Each SM has a dedicated pool of SRAM that it can access
# Since there can be multiple programs per SM, those programs share the same SRAM
    # That will be beneficial information later in the matmul tutorial
TOTAL_SRAM_PER_SM = properties["max_shared_mem"] 
# A warp is a group of threads that execute together
# A thread can be thought of as analogous to a single CPU core but far more limited in the operations it can do
WARP_SIZE = properties["warpSize"]# usually 32 on Nvidia GPUs and 64 on AMD

def triton_softmax(x):
    '''
    helper/wrapper function to 
        1) allocate the output tensor and 
        2) enqueue the above kernel with appropriate grid/block sizes
    
    This wrapper function does not connect us to Pytorch's graph, meaning it does not
    support backpropagation. That (as well as a backward pass kernel) is for a future lesson.
    '''
    # This kernel is only built to support matrices; expanding that support is simple but for a later lesson
    assert x.ndim == 2
    n_rows, n_cols = x.shape

    # The block size is the smallest power of 2 greater than the number of columns in x
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # A trick we can use is to ask the compiler to use more threads per row by
    #  increasing the number of warps (`num_warps`) over which each row is distributed.
    # For now these settings are just a heuristic
    # you will see in the next tutorial how to auto-tune this value in a more natural way
    #   so you don't have to come up with manual heuristics yourself
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    # Rather than executing all code within a kernel sequentially, the GPU can do multiple things simultaneously.
    # This is called the number of software pipelining stages.
    # For example, with two stages, we can have one operate while the other loads the next operands 
    #  from DRAM into SRAM. With 3, we can have one do current operations, load the next operands, and save 
    #  previous operands.
    # Triton needs a number of stages, and it'll handle how to use them efficiently.
    # Here, we use a simple heuristic of "if we've got a lot of memory, use 4. otherwise, use 2"
    num_stages = 4 if TOTAL_SRAM_PER_SM > 200_000 else 2

    # allocate output
    y = torch.empty_like(x)

    # .warmup() pre-compiles kernel and tells us how many registers and how much-shared memory it needs
    kernel = _softmax_kernel.warmup(x, y, # this warm up depends on the attributes of the input and output
                                    x.stride(0), y.stride(0), # see below
                                    n_rows, n_cols,
                                    BLOCK_SIZE=BLOCK_SIZE,
                                    num_stages=num_stages,
                                    num_warps=num_warps,
                                    grid=(1,))
    # x.stride() for each dimension tells us how many entries in memory a pointer needs to move forward in order
    #  to get to the next element of the tensor along the specified dimension. 
    # For any tensor x that is "contiguous", meaning ~cleanly/simply~ defined in memory and for a shape (M, N, K) 
    #  you can expect x.shape(0) == N*K, x.shape(1)==K, and x.shape(2)==1, or more generally 
    #  x.shape(-Z)==math.prod(x.shape[-Z:])
    # A tensor might be non-contiguous if, for example, it's been saved to memory using torch.view() or some similar
    #  operation that leaves the original data in place but messes with dimensions

    # here's the info that the warmup process gave us
    kernel._init_handles()
    n_regs = kernel.n_regs
    sram_needed_per_program = kernel.metadata.shared 

    # and here's how we use that info to set our kernel
    # register-based occupancy
    reg_occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        # each SM has NUM_REGS registers (eg, 65536)
        # Each program uses
            # n_regs per register thread (eg, 32)
            # WARP_SIZE threads per warp (32 on Nvidia, 64 on AMD)
            # num_warps warps per program (4, 8, or 16 in our case with the heuristic above)
        # so each program needs n_regs * WARP_SIZE * num_warps registers total
        # therefore we can fit reg_occupancy programs per SM
        # ex. 65536 // (32 * 32 * 8) = 8 programs per SM (assuming num_warps=8)
    # shared memory-based occupancy
    sram_occupancy = TOTAL_SRAM_PER_SM // sram_needed_per_program
    # determines how many programs can run per SM based on register usage and shared memory usage
    programs_per_sm = min(reg_occupancy, sram_occupancy)
        # The former is the optimal allocation, assuming we have more than enough SRAM
        # the latter is our limit on SRAM when splitting it equally among all SMs
    # Then, given our number of SMs, we calculate how many programs to run in total
    num_programs = min(NUM_SM * programs_per_sm, n_rows)
        # ofc we have another limit since we've got no need to surpass the n_rows in the matrix

    # grid configuration; each row gets its program
    grid = (num_programs, 1, 1)
        # the extra 1's are usually not necessary if they're not being used
        # We use them here because the .warmup() we used earlier has a weird quirk in the way
        #  It's implemented that forces only 3D launch grids to be inputted once it's been used
        # in future lessons we don't use .warmup() so we'll not be required to do this again

    # And now we get to run the kernel with our heuristics-based launch grid
    kernel[grid](
        x, y,
        x.stride(0), y.stride(0),
        n_rows, n_cols,
    )
    return y
