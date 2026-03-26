__const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_MIRRORED_REPEAT | CLK_FILTER_NEAREST;

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

__kernel void SSIM2D(
        __read_only image2d_t input1,
        __read_only image2d_t input2,
        __write_only image2d_t output,
        __private const int windowSizeX,
        __private const int windowSizeY,
        __constant float* weights,
        __private const float c1,
        __private const float c2
    ) {
    const int2 pos = {get_global_id(0), get_global_id(1)};

    const int2 windowSize = {windowSizeX, windowSizeY};
    const int2 halfSize = (windowSize-1)/2;
    float4 mean1 = 0.0f;
    float4 mean2 = 0.0f;
    float4 covariance = 0.0f;
    float4 variance1 = 0.0f;
    float4 variance2 = 0.0f;
    for(int b = -halfSize.y; b <= halfSize.y; ++b) {
        for(int a = -halfSize.x; a <= halfSize.x; ++a) {
            const int2 npos = pos + (int2)(a, b);
            const float4 value1 = readImageAsFloat2D(input1, sampler, npos);
            const float4 value2 = readImageAsFloat2D(input2, sampler, npos);
            const float weight = weights[a+halfSize.x + (b+halfSize.y)*windowSize.x];
            mean1 += value1*weight;
            mean2 += value2*weight;
            covariance += weight*value1*value2;
            variance1 += weight*value1*value1;
            variance2 += weight*value2*value2;
        }
    }
    covariance = covariance - mean1*mean2;
    variance1 = variance1 - mean1*mean1;
    variance2 = variance2 - mean2*mean2;

    float4 luminance = (2.0f*mean1*mean2 + c1) / (mean1*mean1 + mean2*mean2 + c1);
    float4 contrastStructure = (2.0f*covariance + c2) / ( variance1  + variance2 + c2);
    write_imagef(output, pos, luminance*contrastStructure);
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

inline float4 calculateSSIM3D(
        __read_only image3d_t input1,
        __read_only image3d_t input2,
        const int4 pos,
        int windowSizeX,
        int windowSizeY,
        int windowSizeZ,
        __constant float* weights,
        const float c1,
        const float c2
    ) {

    const int3 windowSize = {windowSizeX, windowSizeY, windowSizeZ};
    const int3 halfSize = (windowSize-1)/2;
    float4 mean1 = 0.0f;
    float4 mean2 = 0.0f;
    float4 covariance = 0.0f;
    float4 variance1 = 0.0f;
    float4 variance2 = 0.0f;
    for(int c = -halfSize.z; c <= halfSize.z; ++c) {
        for(int b = -halfSize.y; b <= halfSize.y; ++b) {
            for(int a = -halfSize.x; a <= halfSize.x; ++a) {
                const int4 npos = pos + (int4)(a, b, c, 0);
                const float4 value1 = readImageAsFloat3D(input1, sampler, npos);
                const float4 value2 = readImageAsFloat3D(input2, sampler, npos);
                const float weight = weights[a+halfSize.x + (b+halfSize.y)*windowSize.x + (c+halfSize.z)*windowSize.x*windowSize.y];
                mean1 += value1*weight;
                mean2 += value2*weight;
                covariance += weight*value1*value2;
                variance1 += weight*value1*value1;
                variance2 += weight*value2*value2;
            }
        }
    }
    covariance = covariance - mean1*mean2;
    variance1 = variance1 - mean1*mean1;
    variance2 = variance2 - mean2*mean2;

    float4 luminance = (2.0f*mean1*mean2 + c1) / (mean1*mean1 + mean2*mean2 + c1);
    float4 contrastStructure = (2.0f*covariance + c2) / ( variance1  + variance2 + c2);

    return luminance*contrastStructure;
}

#ifdef fast_3d_image_writes
__kernel void SSIM3D(
        __read_only image3d_t input1,
        __read_only image3d_t input2,
        __write_only image3d_t output,
        __private const int windowSizeX,
        __private const int windowSizeY,
        __private const int windowSizeZ,
        __constant float* weights,
        __private const float c1,
        __private const float c2
    ) {
    const int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};

    write_imagef(output, pos, calculateSSIM3D(input1, input2, pos, windowSizeX, windowSizeY, windowSizeZ, weights, c1, c2));
}
#else
__kernel void SSIM3DBuffer(
        __read_only image3d_t input1,
        __read_only image3d_t input2,
        __global float* output,
        __private const int windowSizeX,
        __private const int windowSizeY,
        __private const int windowSizeZ,
        __constant float* weights,
        __private const float c1,
        __private const float c2,
        __private const int channels
    ) {
    const int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};

    float4 SSIM = calculateSSIM3D(input1, input2, pos, windowSizeX, windowSizeY, windowSizeZ, weights, c1, c2);
    output[(pos.x + pos.y*get_global_size(0) + pos.z*get_global_size(0)*get_global_size(1))*channels] = SSIM.x;
    if(channels > 1)
        output[(pos.x + pos.y*get_global_size(0) + pos.z*get_global_size(0)*get_global_size(1))*channels + 1] = SSIM.y;
    if(channels > 2)
        output[(pos.x + pos.y*get_global_size(0) + pos.z*get_global_size(0)*get_global_size(1))*channels + 2] = SSIM.z;
    if(channels > 3)
        output[(pos.x + pos.y*get_global_size(0) + pos.z*get_global_size(0)*get_global_size(1))*channels + 3] = SSIM.w;
}
#endif