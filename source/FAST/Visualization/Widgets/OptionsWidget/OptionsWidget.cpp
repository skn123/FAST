#include "OptionsWidget.hpp"
#include <QComboBox>
#include <QLabel>
#include <QVBoxLayout>
#include <FAST/Visualization/Window.hpp>
#include <QStylePainter>
#include <utility>
#include <QPushButton>
#include <QButtonGroup>
#include <QGroupBox>

namespace fast {

class ComboBox : public QComboBox {
    protected:
        void paintEvent(QPaintEvent *e) override {
            if(placeholderText().isEmpty()) {
                // If placeholder is empty just use QComboBox paint
                QComboBox::paintEvent(e);
            } else {
                // Override paintEvent for QComboBox due to a bug not show placeholder text
                auto painter = new QStylePainter(this);
                painter->setPen(palette().color(QPalette::Text));
                QStyleOptionComboBox opt;
                initStyleOption(&opt);
                painter->drawComplexControl(QStyle::CC_ComboBox, opt);
                if(currentIndex() < 0) { // Invalid index
                    opt.palette.setBrush(QPalette::ButtonText,opt.palette.brush(QPalette::ButtonText).color().lighter());
                    opt.currentText = placeholderText();
                }
                painter->drawControl(QStyle::CE_ComboBoxLabel, opt);
                painter->end();
            }
        };
};
OptionsWidget::OptionsWidget(const std::vector<std::string> &options, const std::string &name, Type type, const std::string& placeholder, int selected,
                           OptionsWidgetCallback *callback, QWidget *parent) : Widget(parent) {
    init(type, name, placeholder, options, selected);
    m_callbackClass = callback;
}

OptionsWidget::OptionsWidget(const std::vector<std::string> &options, const std::string &name, Type type, const std::string& placeholder, int selected,
                           std::function<void(int, std::string)> callback, QWidget *parent) : Widget(parent) {
    init(type, name, placeholder, options, selected);
    m_callbackFunction = std::move(callback);
}

void OptionsWidget::init(Type type, const std::string& name, const std::string& placeholder, const std::vector<std::string>& options, int selected) {
    m_type = type;
    m_name = name;
    m_options = options;
    auto layout = new QVBoxLayout();
    setLayout(layout);
    if(type == DROPDOWN) {
        if(!name.empty()) {
            m_label = new QLabel();
            m_label->setText(QString::fromStdString(name));
            layout->addWidget(m_label);
        }
        m_comboBox = new ComboBox();
        layout->addWidget(m_comboBox);
        if(!placeholder.empty()) {
            m_comboBox->setPlaceholderText(QString::fromStdString(placeholder));
        }
        for(auto& option : options)
            m_comboBox->addItem(QString::fromStdString(option));

        if(!placeholder.empty()) {
            setSelected(selected);
        } else {
            if(selected < 0) {
                setSelected(0);
            } else {
                setSelected(selected);
            }
        }
        QObject::connect(m_comboBox, &QComboBox::currentTextChanged, [=](const QString& value) {
            auto text = value.toStdString();
            auto index = m_comboBox->findText(value);
            if(m_callbackClass != nullptr) {
                m_callbackClass->handle(index, text);
            } else {
                m_callbackFunction(index, text);
            }
        });
    } else if(type == BUTTONS) {
        auto box = new QGroupBox(QString::fromStdString(name), this);
        auto boxLayout = new QVBoxLayout(box);
        m_buttonGroup = new QButtonGroup(this);
        m_buttonGroup->setExclusive(true);
        int i = 0;
        for(auto& option : options) {
            auto button = new QPushButton(QString::fromStdString(option), box);
            button->setCheckable(true);
            boxLayout->addWidget(button);
            m_buttonGroup->addButton(button, i);
            ++i;
        }
        layout->addWidget(box);
        if(selected >= 0)
            setSelected(selected);
        QObject::connect(m_buttonGroup, &QButtonGroup::idClicked, [=](int index) {
            auto text = m_options[index];
            if(m_callbackClass != nullptr) {
                m_callbackClass->handle(index, text);
            } else {
                m_callbackFunction(index, text);
            }
        });
    }
}

void OptionsWidget::setSelected(std::string value) {
    // TODO Thread safety?
    if(m_type == DROPDOWN) {
        m_comboBox->setCurrentText(QString::fromStdString(value));
    } else {
        int index = -1;
        for(int i = 0; i < m_options.size(); ++i) {
            if(m_options[i] == value)
                index = i;
        }
        if(index < 0)
            throw Exception("Value " + value + " was not found in options");
        m_buttonGroup->button(index)->setChecked(true);
    }
}

void OptionsWidget::setSelected(int index) {
    // TODO Thread safety?
    if(m_type == DROPDOWN) {
        m_comboBox->setCurrentIndex(index);
    } else {
        m_buttonGroup->button(index)->setChecked(true);
    }
}

int OptionsWidget::getSelected() {
    if(m_type == DROPDOWN) {
        return m_comboBox->currentIndex();
    } else {
        return m_buttonGroup->checkedId();
    }
}

std::string OptionsWidget::getOption(int index) const {
    if(index < 0 || index >= m_options.size())
        throw Exception("Index out of range in OptionsWidget::getOption()");
    return m_options.at(index);
}

}