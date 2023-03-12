#pragma once

#include <QWindow>
#include <QOpenGLContext>

class OffscreenBuffer : public QWindow
{
public:
    OffscreenBuffer();

    QOpenGLContext* getContext() { return _context; }

    /** Initialize and bind the OpenGL context associated with this buffer */
    void initialize();

    /** Bind the OpenGL context associated with this buffer */
    void bindContext();

    /** Release the OpenGL context associated with this buffer */
    void releaseContext();

private:
    QOpenGLContext* _context;
};
