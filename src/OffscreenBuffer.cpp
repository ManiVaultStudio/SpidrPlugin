#include "hdi/dimensionality_reduction/gradient_descent_tsne_texture.h" // Included for glad, must be included before OpenGLContext

#include "OffscreenBuffer.h"

OffscreenBuffer::OffscreenBuffer() :
    _context(nullptr)
{
    setSurfaceType(QWindow::OpenGLSurface);

    create();
}

void OffscreenBuffer::initialize()
{
    QOpenGLContext* globalContext = QOpenGLContext::globalShareContext();
    _context = new QOpenGLContext(this);
    _context->setFormat(globalContext->format());

    if (!_context->create())
        qFatal("Cannot create requested OpenGL context.");

    _context->makeCurrent(this);

#ifndef __APPLE__
    if (!gladLoadGL()) {
        qFatal("No OpenGL context is currently bound, therefore OpenGL function loading has failed.");
    }
#endif // Not __APPLE__
}

void OffscreenBuffer::bindContext()
{
    _context->makeCurrent(this);
}

void OffscreenBuffer::releaseContext()
{
    _context->doneCurrent();
}
