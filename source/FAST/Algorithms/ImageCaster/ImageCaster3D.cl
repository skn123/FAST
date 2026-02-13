__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

inline float4 readImageAsFloat3D(__read_only image3d_t image, sampler_t sampler, int4 position) {
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
inline void writeImageAsFloat3D(__write_only image3d_t image, int4 position, float4 value) {
    int dataType = get_image_channel_data_type(image);
    if(dataType == CLK_FLOAT || dataType == CLK_SNORM_INT16 || dataType == CLK_UNORM_INT16) {
        write_imagef(image, position, value);
    } else if(dataType == CLK_SIGNED_INT8 || dataType == CLK_SIGNED_INT16 || dataType == CLK_SIGNED_INT32) {
        write_imagei(image, position, convert_int4(round(value)));
    } else {
        write_imageui(image, position, convert_uint4(round(value)));
    }
}

__kernel void cast3D(
        __read_only image3d_t input,
        __write_only image3d_t output,
        __private float scaleFactor,
        __private char normalize,
        __private float minimum,
        __private float maximum
    ) {
    const int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    if(normalize == 1) {
        float4 value = readImageAsFloat3D(input, sampler, pos);
        value = (value - minimum) / (maximum - minimum);
        writeImageAsFloat3D(output, pos, value*scaleFactor);
    } else {
        writeImageAsFloat3D(output, pos, readImageAsFloat3D(input, sampler, pos)*scaleFactor);
    }
}
#else

inline void writeImageAsFloat3D(
        __global TYPE* output,
        const int4 pos,
        const int2 size,
        const int channels,
        const float4 value
        ) {
    float valuePtr[4] = {value.x, value.y, value.z, value.w};
    for(int i = 0; i < channels; ++i)
        output[(pos.x + pos.y*size.x + pos.z*size.x*size.y)*channels + i] = valuePtr[i];
}

__kernel void cast3DBuffer(
        __read_only image3d_t input,
        __global TYPE* output,
        __private const float scaleFactor,
        __private const char normalize,
        __private const float minimum,
        __private const float maximum,
        __private const int channels
    ) {
    const int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    float4 value = readImageAsFloat3D(input, sampler, pos);
    if(normalize == 1) {
        value = (value - minimum) / (maximum - minimum);
    }
    writeImageAsFloat3D(output, pos, (int2)(get_global_size(0), get_global_size(1)), channels, value*scaleFactor);
}
#endif