__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

#ifdef fast_3d_image_writes
__kernel void gaussianSmoothingSeparable(
        __read_only image3d_t input,
        __constant float * mask,
        __write_only image3d_t output,
        __private int maskSize,
        __private uchar direction
        ) {

    const int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    const int halfSize = (maskSize-1)/2;

    float sum = 0.0f;
    int dataType = get_image_channel_data_type(input);
    for(int i = -halfSize; i <= halfSize; ++i) {
        int4 offset = {0,0,0,0};
        if(direction == 0) {
            offset.x = i;
        } else if(direction == 1) {
            offset.y = i;
        } else {
            offset.z = i;
        }
        const uchar maskOffset = halfSize + i;
        if(dataType == CLK_FLOAT) {
            sum += mask[maskOffset]*read_imagef(input, sampler, pos+offset).x;
        } else if(dataType == CLK_UNSIGNED_INT8 || dataType == CLK_UNSIGNED_INT16 || dataType == CLK_UNSIGNED_INT32) {
            sum += mask[maskOffset]*read_imageui(input, sampler, pos+offset).x;
        } else {
            sum += mask[maskOffset]*read_imagei(input, sampler, pos+offset).x;
        }
    }

    int outputDataType = get_image_channel_data_type(output);
    if(outputDataType == CLK_FLOAT) {
        write_imagef(output, pos, sum);
    } else if(outputDataType == CLK_UNSIGNED_INT8 || outputDataType == CLK_UNSIGNED_INT16 || outputDataType == CLK_UNSIGNED_INT32) {
        write_imageui(output, pos, round(sum));
    } else {
        write_imagei(output, pos, round(sum));
    }
}

#endif
__kernel void gaussianSmoothing(
        __read_only image3d_t input,
        __constant float * mask,
        __global TYPE* output,
        __private int maskSizeX,
        __private int maskSizeY,
        __private int maskSizeZ
        ) {

    const int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    const int3 maskSize = {maskSizeX, maskSizeY, maskSizeZ};
    const int3 halfSize = (maskSize-1)/2;

    float sum = 0.0f;
    const int dataType = get_image_channel_data_type(input);
    for(int z = -halfSize.z; z <= halfSize.z; ++z) {
    for(int y = -halfSize.y; y <= halfSize.y; ++y) {
    for(int x = -halfSize.x; x <= halfSize.x; ++x) {
        const int4 offset = {x,y,z,0};
        const uint maskOffset = x+halfSize.x+(y+halfSize.y)*maskSize.x+(z+halfSize.z)*maskSize.x*maskSize.y;
        if(dataType == CLK_FLOAT) {
            sum += mask[maskOffset]*read_imagef(input, sampler, pos+offset).x;
        } else if(dataType == CLK_UNSIGNED_INT8 || dataType == CLK_UNSIGNED_INT16 || dataType == CLK_UNSIGNED_INT32) {
            sum += mask[maskOffset]*read_imageui(input, sampler, pos+offset).x;
        } else {
            sum += mask[maskOffset]*read_imagei(input, sampler, pos+offset).x;
        }
    }}}

    output[pos.x+pos.y*get_global_size(0)+pos.z*get_global_size(0)*get_global_size(1)] = sum;
}
