#ifndef YYLVV_LINEGLYPH_CUH
#define YYLVV_LINEGLYPH_CUH

#include "../renderstate.cuh"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "../VAO.h"
#include "../Program.h"

class LineGlyphRenderState : public RenderState {
public:
    LineGlyphRenderState();
    virtual void initialize(YYLVVRes &res, UIRes &ui_res) override;
    virtual void destroy() override;
    virtual void render(YYLVVRes &res, UIRes &ui_res) override;
    virtual void process_events(YYLVVRes &res, UIRes &ui_res) override;
    virtual void key_pressed(YYLVVRes &res, UIRes &ui_res, int key) override;

private:
    void initialize_line_glyph_resources();
    bool generate_line_glyphs(YYLVVRes &res, UIRes &ui_res, float z);
    void clamp_visualizing_z(YYLVVRes &res);

    // Vector field line glyph configurations
    int num_lines_x, num_lines_y;
    std::shared_ptr<VAO> line_glyph_vao;
    std::shared_ptr<Program> line_glyph_program;
    cudaGraphicsResource *line_glyph_graphics_resource;
    float vector_length;
    float current_visualizing_z;
    bool visualize_xy;
};

#endif
