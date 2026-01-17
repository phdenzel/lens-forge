// Vertex shader - creates a full-screen quad
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // Full-screen quad using two triangles (6 vertices)
    // Triangle 1: (0,0), (2,0), (0,2)
    // Triangle 2: forms the rest of the quad
    let x = f32((vertex_index & 1u) << 1u);
    let y = f32((vertex_index & 2u));
    
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.tex_coords = vec2<f32>(x, y);
    
    return out;
}

// Fragment shader - samples the texture
@group(0) @binding(0)
var t_texture: texture_2d<f32>;

@group(0) @binding(1)
var t_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_texture, t_sampler, in.tex_coords);
}
