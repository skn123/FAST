#pragma once

#include <QWidget>
#include <QVariant>
#include <FASTExport.hpp>

namespace fast {
/**
 * @brief Abstract base class of widgets
 * @ingroup widgets
 */
class FAST_EXPORT Widget : public QWidget {
    public:
        /**
         * @brief Set Qt property for this widget.
         * This is just a wrapper for QObject::setProperty()
         * @tparam T
         * @param name
         * @param value
         */
        template <class T>
        void setProperty(const std::string& name, T value);
        /**
         * @brief Set Qt object name for this widget.
         * This is just a wrapper for QObject::setObjectName()
         * @param name
         */
        void setObjectName(const std::string& name);
    protected:
        using QWidget::QWidget;
};

template<class T>
void Widget::setProperty(const std::string& name, T value) {
    QObject::setProperty(name.c_str(), QVariant(value));
}

template<>
void Widget::setProperty(const std::string &name, std::string value);

#ifdef SWIG
%template(setPropertyInt) Widget::setProperty<int>;
%template(setPropertyFloat) Widget::setProperty<float>;
%template(setPropertyString) Widget::setProperty<std::string>;

%extend Widget {
    %pythoncode %{
        def setProperty(self, name:str, value:Union[int,float,str]):
            """
            Set Qt property for this widget.
            This is just a wrapper for QObject::setProperty()
            """
            if isinstance(value, int):
                return self.setPropertyInt(name, value)
            elif isinstance(value, float):
                return self.setPropertyFloat(name, value)
            elif isinstance(value, str):
                return self.setPropertyString(name, value)
            else:
                raise TypeError(f"Unsupported type in Widget::setProperty: {type(value)}")
    %}
}
#endif

}