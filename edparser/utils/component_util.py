# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-31 19:24
import os
import traceback
from sys import exit

from edparser import pretrained
from edparser.common.component import Component
from edparser.utils.io_util import get_resource, load_json, eprint
from edparser.utils.reflection import object_from_class_path, str_to_type
from edparser import version


def load_from_meta_file(save_dir: str, meta_filename='meta.json', transform_only=False, load_kwargs=None,
                        **kwargs) -> Component:
    identifier = save_dir
    load_path = save_dir
    save_dir = get_resource(save_dir)
    if save_dir.endswith('.json'):
        meta_filename = os.path.basename(save_dir)
        save_dir = os.path.dirname(save_dir)
    metapath = os.path.join(save_dir, meta_filename)
    if not os.path.isfile(metapath):
        tips = ''
        if save_dir.isupper():
            from difflib import SequenceMatcher
            similar_keys = sorted(pretrained.ALL.keys(),
                                  key=lambda k: SequenceMatcher(None, save_dir, metapath).ratio(),
                                  reverse=True)[:5]
            tips = f'Check its spelling based on the available keys:\n' + \
                   f'{sorted(pretrained.ALL.keys())}\n' + \
                   f'Tips: it might be one of {similar_keys}'
        raise FileNotFoundError(f'The identifier {save_dir} resolves to a non-exist meta file {metapath}. {tips}')
    meta: dict = load_json(metapath)
    cls = meta.get('class_path', None)
    assert cls, f'{meta_filename} doesn\'t contain class_path field'
    try:
        obj: Component = object_from_class_path(cls, **kwargs)
        if hasattr(obj, 'load'):
            if transform_only:
                # noinspection PyUnresolvedReferences
                obj.load_transform(save_dir)
            else:
                if load_kwargs is None:
                    load_kwargs = {}
                if os.path.isfile(os.path.join(save_dir, 'config.json')):
                    obj.load(save_dir, **load_kwargs)
                else:
                    obj.load(metapath, **load_kwargs)
            obj.meta['load_path'] = load_path
        return obj
    except Exception as e:
        eprint(f'Failed to load {identifier}. See stack trace below')
        traceback.print_exc()
        model_version = meta.get("hanlp_version", "unknown")
        cur_version = version.__version__
        if model_version != cur_version:
            eprint(
                f'{identifier} was created with hanlp-{model_version}, while you are running {cur_version}. '
                f'Try to upgrade hanlp with\n'
                f'pip install --upgrade hanlp\n'
                f'If the problem persists, please submit an issue to https://github.com/hankcs/HanLP/issues .')
        exit(1)


def load_from_meta(meta: dict) -> Component:
    cls = meta.get('class_path', None)
    assert cls, f'{meta} doesn\'t contain class_path field'
    cls = str_to_type(cls)
    return cls.from_meta(meta)
