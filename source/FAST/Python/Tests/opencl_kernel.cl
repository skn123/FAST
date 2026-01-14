__const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_NONE;
__kernel void invert(
    __read_only image2d_t input,
    __write_only image2d_t output,
    __private int max
    ) {
    int2 pos = {get_global_id(0), get_global_id(1)};
    int value = read_imageui(input, sampler, pos).x;
    write_imageui(output, pos, max - value);
}
