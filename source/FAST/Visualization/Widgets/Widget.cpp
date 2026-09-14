#include "Widget.hpp"

namespace fast {

void Widget::setObjectName(const std::string& name) {
    QObject::setObjectName(QString::fromStdString(name));
}

template<>
void Widget::setProperty(const std::string &name, std::string value) {
    QObject::setProperty(name.c_str(), value.c_str());
}

}