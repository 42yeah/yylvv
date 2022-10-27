#ifndef YYLVV_ARROWGLYPH_CUH
#define YYLVV_ARROWGLYPH_CUH

#include "../renderstate.cuh"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "../VAO.h"
#include "../Program.h"

class ArrowGlyphRenderState : public RenderState {
public:
    ArrowGlyphRenderState();
    virtual void initialize(YYLVVRes &res, UIRes &ui_res) override;
    virtual void destroy() override;
    virtual void render(YYLVVRes &res, UIRes &ui_res) override;
    virtual void process_events(YYLVVRes &res, UIRes &ui_res) override;
    virtual void key_pressed(YYLVVRes &res, UIRes &ui_res, int key) override;

private:
    void initialize_arrow_glyph_resources();
    bool generate_arrow_glyphs(YYLVVRes &res, UIRes &ui_res, float z);
    void clamp_visualizing_z(YYLVVRes &res);

    // Vector field arrow glyph configurations
    int num_arrows_x, num_arrows_y;
    std::shared_ptr<VAO> arrow_glyph_vao;
    std::shared_ptr<Program> arrow_glyph_program;
    cudaGraphicsResource *arrow_glyph_graphics_resource;
    float vector_length;
    float current_visualizing_z;
    bool visualize_xy;
};

#endif
