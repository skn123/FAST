__const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

float4 readImageAsFloat2D(__read_only image2d_t image, sampler_t sampler, int2 position) {
    int dataType = get_image_channel_data_type(image);
    if(dataType == CLK_FLOAT || dataType == CLK_SNORM_INT16 || dataType == CLK_UNORM_INT16) {
        return read_imagef(image, sampler, position);
    } else if(dataType == CLK_SIGNED_INT8 || dataType == CLK_SIGNED_INT16 || dataType == CLK_SIGNED_INT32) {
        return convert_float4(read_imagei(image, sampler, position));
    } else {
        return convert_float4(read_imageui(image, sampler, position));
    }
}

__kernel void squaredError2D(
        __read_only image2d_t input1,
        __read_only image2d_t input2,
        __write_only image2d_t output
    ) {
    const int2 pos = {get_global_id(0), get_global_id(1)};
    float4 value1 = readImageAsFloat2D(input1, sampler, pos);
    float4 value2 = readImageAsFloat2D(input2, sampler, pos);
    write_imagef(output, pos, (value1 - value2)*(value1 - value2));
}

float4 readImageAsFloat3D(__read_only image3d_t image, sampler_t sampler, int4 position) {
    int dataType = get_image_channel_data_type(image);
    if(dataType == CLK_FLOAT || dataType == CLK_SNORM_INT16 || dataType == CLK_UNORM_INT16) {
        return read_imagef(image, sampler, position);
    } else if(dataType == CLK_SIGNED_INT8 || dataType == CLK_SIGNED_INT16 || dataType == CLK_SIGNED_INT32) {
        return convert_float4(read_imagei(image, sampler, position));
    } else {
        return convert_float4(read_imageui(image, sampler, position));
    }
}

#ifdef fast_3d_image_writes
__kernel void squaredError3D(
        __read_only image3d_t input1,
        __read_only image3d_t input2,
        __write_only image3d_t output
    ) {
    const int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    float4 value1 = readImageAsFloat3D(input1, sampler, pos);
    float4 value2 = readImageAsFloat3D(input2, sampler, pos);
    write_imagef(output, pos, (value1 - value2)*(value1 - value2));
}
#else
__kernel void squaredError3DBuffer(
        __read_only image3d_t input1,
        __read_only image3d_t input2,
        __global float* output,
        __private const int channels
    ) {
    const int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    float4 value1 = readImageAsFloat3D(input1, sampler, pos);
    float4 value2 = readImageAsFloat3D(input2, sampler, pos);
    float4 squaredError = (value1 - value2)*(value1 - value2);
    output[(pos.x + pos.y*get_global_size(0) + pos.z*get_global_size(0)*get_global_size(1))*channels] = squaredError.x;
    if(channels > 1)
        output[(pos.x + pos.y*get_global_size(0) + pos.z*get_global_size(0)*get_global_size(1))*channels + 1] = squaredError.y;
    if(channels > 2)
        output[(pos.x + pos.y*get_global_size(0) + pos.z*get_global_size(0)*get_global_size(1))*channels + 2] = squaredError.z;
    if(channels > 3)
        output[(pos.x + pos.y*get_global_size(0) + pos.z*get_global_size(0)*get_global_size(1))*channels + 3] = squaredError.w;
}
#endif