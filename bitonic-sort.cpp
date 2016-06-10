#define __CL_ENABLE_EXCEPTIONS
#define PROGRAM_FILE                "bitonic-sort.cl"
#define BITONIC_SORT_INIT           "bitonic_sort_init"
#define BITONIC_SORT_STAGE_0        "bitonic_sort_stage_zero"
#define BITONIC_SORT_STAGE_N        "bitonic_sort_stage_n"
#define BITONIC_SORT_MERGE          "bitonic_sort_merge"
#define BITONIC_SORT_MERGE_LAST     "bitonic_sort_merge_last"
#define DIRECTION                   0
#define DATA_SIZE                   32
#define PRESENT_DATA_OUTPUT         false
#define PRESENT_DATA_INPUT          false

#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>
#include <random>
#include <CL/cl.hpp>

cl_int chech_integrity(int *_data);
void init_data(int *_data)
{
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(1,100);

    for(int i = 0; i < DATA_SIZE; ++i)
    {
        _data[i] = distribution(generator);
        #if PRESENT_DATA_INPUT
            std::cout << _data[i] << std::endl;
        #endif
    }
}

int main(int argc, char const *argv[])
{

    int data[DATA_SIZE]; init_data(data);

    size_t local_size, global_size;
    cl_uint stage, high_stage, num_stages;
    cl_int i, err, check, direction;

    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> platform_devices, ctx_devices;
    std::vector<std::string> device_names;
    std::vector<size_t> device_max_work_item_sizes;

    cl_int a = cl::Platform::get(&platforms);
    platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &platform_devices);

    std::cout << "Your devices are: \n\n";
    for(int i = 0; i < platform_devices.size(); ++i)
    {
        std::cout << platform_devices[i].getInfo<CL_DEVICE_VENDOR>() << std::endl;
        std::cout << platform_devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
        std::cout << "Max compute units: ";
        std::cout << platform_devices[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
        std::cout << "Max work item dimensions: ";
        std::cout << platform_devices[i].getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() << std::endl;

        device_max_work_item_sizes = platform_devices[i].getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
        std::cout << "Max work item sizes are: ";
        for_each(device_max_work_item_sizes.begin(), device_max_work_item_sizes.end(),
            [&](size_t _size)
            {
                std::cout << _size << " ";
            }
        );
        std::cout << "\n\n";
    }

    cl::Context context(platform_devices);
    ctx_devices = context.getInfo<CL_CONTEXT_DEVICES>();

    for(int i = 0; i < ctx_devices.size(); ++i)
        std::cout << '[' << i << ']'
            << ctx_devices[i].getInfo<CL_DEVICE_NAME>().c_str() << std::endl;

    /* Open and build program */
    std::ifstream program_file(PROGRAM_FILE);

    std::string program_string(std::istreambuf_iterator<char>(program_file),
        (std::istreambuf_iterator<char>()));

    cl::Program::Sources source(1, std::make_pair(program_string.c_str(),
        program_string.length()+1));

    cl::Program program(context, source);
    program.build(platform_devices);

    //creating kernels

    cl::Kernel kernel_init(program, BITONIC_SORT_INIT);
    cl::Kernel kernel_stage_0(program, BITONIC_SORT_STAGE_0);
    cl::Kernel kernel_stage_n(program, BITONIC_SORT_STAGE_N);
    cl::Kernel kernel_merge(program, BITONIC_SORT_MERGE);
    cl::Kernel kernel_merge_last(program, BITONIC_SORT_MERGE_LAST);

    std::vector<cl::Kernel *> kernels = { &kernel_init, &kernel_stage_0,
        &kernel_stage_n, &kernel_merge, &kernel_merge_last };

    /* Determine maximum work-group size */
    local_size = kernel_init.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(ctx_devices[0]);

    local_size = (int)pow(2, trunc(log2(local_size)));

    /* Create buffer */
    cl::Buffer data_buffer(context,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(data), data);

    for(int i = 0; i < kernels.size(); ++i)
    {
        (*kernels[i]).setArg(0, data_buffer);
        (*kernels[i]).setArg(1, 8*local_size*sizeof(float), NULL);
    }

    /* Create a command queue with SPECIFIC device!*/
    cl::CommandQueue queue(context, ctx_devices[0], 0, nullptr);

    global_size = DATA_SIZE / 8;

    if(global_size < local_size)
        local_size = global_size;

    queue.enqueueNDRangeKernel(kernel_init, 1, global_size, local_size);

    /* Execute further stages */
    num_stages = global_size / local_size;

    for(high_stage = 2; high_stage < num_stages; high_stage <<= 1)
    {

        kernel_stage_0.setArg(2, sizeof(int), &high_stage);
        kernel_stage_n.setArg(3, sizeof(int), &high_stage);

        for(stage = high_stage; stage > 1; stage >>= 1)
        {

            kernel_stage_n.setArg(2, sizeof(int), &stage);

            queue.enqueueNDRangeKernel(
                kernel_stage_n, 1, global_size, local_size);

        }

        queue.enqueueNDRangeKernel(
            kernel_stage_0, 1, global_size, local_size);

    }

    /* Set the sort direction */
    direction = DIRECTION;
    kernel_merge.setArg(3, sizeof(int), &direction);
    kernel_merge_last.setArg(2, sizeof(int), &direction);

    /* Perform the bitonic merge */
    for(stage = num_stages; stage > 1; stage >>= 1)
    {

        kernel_merge.setArg(2, sizeof(int), &stage);

        queue.enqueueNDRangeKernel(
            kernel_merge, 1, global_size, local_size);

    }

    queue.enqueueNDRangeKernel(
        kernel_merge_last, 1, global_size, local_size);

    /* Read the result */
    queue.enqueueReadBuffer(
        data_buffer, CL_TRUE, 0, sizeof(data), data, NULL, NULL
    );

    if(chech_integrity(data))
    {
        std::cout << "Success!" << std::endl;
        if(PRESENT_DATA_OUTPUT)
            for(int i = 0; i < DATA_SIZE; ++i)
                std::cout << data[i] << std::endl;

    }
    else
    {
        std::cout << "Sotring failed." << std::endl;
    }


    return 0;

}

cl_int chech_integrity(int *_data)
{

    for(cl_int i = 1; i < DATA_SIZE; i++)
        if(_data[i]

            #if DIRECTION == 0
            <
            #else
            >
            #endif

             _data[i-1])
        {
            return 0;
            break;
        }

    return 1;

}
