#ifndef YYLVV_RENDERSTATE_CUH
#define YYLVV_RENDERSTATE_CUH

#include "yylvv.cuh"

struct UIRes;
struct YYLVVRes;

class RenderState {
public:
    virtual void initialize(YYLVVRes &res, UIRes &ui_res) = 0;
    virtual void destroy() = 0;
    virtual void render(YYLVVRes &res, UIRes &ui_res) = 0;
    virtual void process_events(YYLVVRes &res, UIRes &ui_res) = 0;
    virtual void key_pressed(YYLVVRes &res, UIRes &ui_res, int key) = 0;
};

#endif
