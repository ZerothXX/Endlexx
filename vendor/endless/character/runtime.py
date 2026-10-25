"""Shared character selection for both public command-line entries."""
from pathlib import Path
import argparse

import config


def available_characters():
    """列出可作为训练/推理目标的角色目录。

    只接受含 images/ 子目录的角色（数据准备后统一落到 images/），
    避免把尚未整理、没有 images/ 的目录列进菜单（选中即崩）。
    """
    root = Path(config.DATASET_DIR)
    if not root.is_dir():
        return []
    extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    names = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        images = entry / 'images'
        if not images.is_dir():
            continue
        if any(f.is_file() and f.suffix.lower() in extensions for f in images.iterdir()):
            names.append(entry.name)
    return sorted(names)


def select_character(character=None, *, interactive=False):
    names = available_characters()
    if not names:
        raise RuntimeError('dataset 下没有含图片的角色目录。')
    if interactive and character is None:
        print('请选择本次使用的角色：')
        for index, name in enumerate(names, 1):
            print(f'  [{index}] {name}')
        while True:
            try:
                answer = input('输入角色名或序号（q 退出）: ').strip()
            except EOFError as exc:
                raise RuntimeError('无交互输入；请通过 --character 角色名明确选择。') from exc
            if answer.lower() == 'q':
                raise SystemExit(0)
            # Exact names take priority: a numeric character ID is not an index.
            if answer in names:
                character = answer
                break
            if answer.isdigit() and 1 <= int(answer) <= len(names):
                character = names[int(answer) - 1]
                break
            print('选择无效，请重新输入。')
    character = str(character if character is not None else config.CHARACTER_ID)
    if character not in names:
        raise ValueError(f'角色 {character!r} 不存在或没有图片；可用角色：{names}')
    config.CHARACTER_ID = character
    config.TRIGGER_WORD = config.get_trigger(character)
    print(f'[角色] {character}；触发词 {config.TRIGGER_WORD}')
    return character


def select_cli_character(argv=None, *, allow_inputs=False, allow_stage=False):
    parser = argparse.ArgumentParser(description='从 dataset 中选择目标角色')
    parser.add_argument('--character', help='角色目录名；省略时显示选择菜单')
    if allow_inputs:
        # argparse 的 nargs 写成字符串 "+" 时不会做列表展开：它会把字面量 "+" 当成第一个
        # 文件塞进 args.input，于是 "--input <目录>" 变成 ["+", "<目录>"]，
        # 目录路径被当成图片文件校验后直接报错（实测）。这里必须用真正的 nargs="+"。
        parser.add_argument('--input', nargs='+', help='指定本次测试图片；省略时使用配置输入目录')
    if allow_stage:
        parser.add_argument('--stage', choices=('all', 'identity', 'concept'), default='all',
                            help='all=身份+概念（默认）；identity=只训身份；concept=只训概念分支')
    args = parser.parse_args(argv)
    if allow_inputs and args.input:
        config.INPUT_IMAGE = args.input
    if allow_stage:
        config.TRAIN_STAGE = args.stage
    return select_character(args.character, interactive=args.character is None)
