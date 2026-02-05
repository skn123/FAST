__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void gaussianSmoothing(
        __read_only image2d_t input,
        __constant float * mask,
        __write_only image2d_t output,
        __private int maskSizeX,
        __private int maskSizeY
        ) {

    const int2 pos = {get_global_id(0), get_global_id(1)};
    const int2 maskSize = {maskSizeX, maskSizeY};
    const int2 halfSize = (maskSize-1)/2;

    float sum = 0.0f;
    int dataType = get_image_channel_data_type(input);
    for(int x = -halfSize.x; x <= halfSize.x; ++x) {
    for(int y = -halfSize.y; y <= halfSize.y; ++y) {
        const int2 offset = {x,y};
        if(dataType == CLK_FLOAT) {
            sum += mask[x+halfSize.x+(y+halfSize.x)*maskSize.x]*read_imagef(input, sampler, pos+offset).x;
        } else if(dataType == CLK_UNSIGNED_INT8 || dataType == CLK_UNSIGNED_INT16 || dataType == CLK_UNSIGNED_INT32) {
            sum += mask[x+halfSize.x+(y+halfSize.y)*maskSize.x]*read_imageui(input, sampler, pos+offset).x;
        } else {
            sum += mask[x+halfSize.x+(y+halfSize.y)*maskSize.x]*read_imagei(input, sampler, pos+offset).x;
        }
    }}

    int outputDataType = get_image_channel_data_type(output);
    if(outputDataType == CLK_FLOAT) {
        write_imagef(output, pos, sum);
    } else if(outputDataType == CLK_UNSIGNED_INT8 || outputDataType == CLK_UNSIGNED_INT16 || outputDataType == CLK_UNSIGNED_INT32) {
        write_imageui(output, pos, round(sum));
    } else {
        write_imagei(output, pos, round(sum));
    }
}
