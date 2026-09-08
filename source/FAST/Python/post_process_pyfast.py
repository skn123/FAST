import sys
try:
    sys.path.insert(0, '.')
    import fast
except Exception as e:
    print('Failed to import fast')
    exit()
import re
import shutil
import inspect

classes = inspect.getmembers(fast, inspect.isclass)
print(f'Found total of {len(classes)} classes')

target_classes = {}
for name, cls in classes:
    is_fast_object = issubclass(cls, fast.Object)
    has_create_method = hasattr(cls, 'create') and callable(getattr(cls, 'create'))
    #print(name, is_fast_object, has_create_method)

    if is_fast_object and has_create_method:
        create_method_is_static = '@staticmethod' in inspect.getsource(cls.create)
        #print('Is static?', create_method_is_static)
        if create_method_is_static:
            create_method_source = inspect.getsource(cls.create).replace('@staticmethod', '')
            new_method = create_method_source.replace('def create(', 'def __new__(cls, ')
            init_method = create_method_source.replace('def create(', 'def __init__(self, ')
            init_method = re.sub( f'return _fast.{name}_create(.*)', 'pass', init_method)

            target_classes[name] = {
                'name': name,
                'cls': cls,
                'init_source': init_method,
                'new_source':  new_method,
            }

# Second pass
for name, cls in classes:
    if name in target_classes:
        create_method_source = inspect.getsource(cls.create).replace('@staticmethod', '')
        call_method_source = create_method_source \
            .replace('def create(', 'def __call__(cls, ') \
            .replace(f'_fast.{name}_create(', 'cls.create(')
        parent = cls.__base__.__name__
        metaclass_parent = 'type'
        if parent in target_classes:
            metaclass_parent = parent + 'MetaClass'
        metaclass = f'class {name}MetaClass({metaclass_parent}):\n{call_method_source}\n\n'
        target_classes[name]['metaclass'] = metaclass

print(f'Found {len(target_classes)} classes to modify.')

input_file = open('fast/fast.py', 'r', encoding='utf-8')

processClass = False
lines_for_new_file = []
skip_lines = 0
for line in input_file:
    if skip_lines > 0:
        skip_lines -= 1
        continue
    if line.startswith('class '):
        currentClassName = line[6:line.find('(')]
        processClass = False
        if currentClassName in target_classes:
            processClass = True
            for x in target_classes[currentClassName]['metaclass'].split('\n'):
                lines_for_new_file.append(x + '\n')
            line = line.replace('):', f',metaclass={currentClassName}MetaClass):')
    if processClass:
        if line.strip().startswith('def create('):
            staticmethod_line = lines_for_new_file.pop() # Remove @staticmethod

            # Append __init__
            for x in target_classes[currentClassName]['init_source'].split('\n'):
                lines_for_new_file.append(x + '\n')

            lines_for_new_file.append(staticmethod_line)
        if line.strip().startswith('def __init__'): # Skip old __init__
            skip_lines = 1
            continue

    lines_for_new_file.append(line)

input_file.close()

if not line.startswith('# Post processing done'):
    lines_for_new_file.append('# Post processing done.')
    output_file = open('fast/fast_post_processed.py', 'w', encoding='utf-8')
    output_file.writelines(lines_for_new_file)
    output_file.close()
    shutil.move('fast/fast_post_processed.py', 'fast/fast.py')
    print('Wrote post processed file.')
else:
    print('File already post processed. Skipping.')
