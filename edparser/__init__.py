# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-06-13 18:05
import os

if not os.environ.get('HANLP_SHOW_TF_LOG', None):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
    import absl.logging, logging

    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.root.removeHandler(absl.logging._absl_handler)
    exec('absl.logging._warn_preinit_stderr = False')  # prevent exporting _warn_preinit_stderr
from edparser.utils.tf_util import set_gpu_memory_growth, set_gpu
from edparser.utils.util import ls_resource_in_module
import edparser.callbacks
import edparser.common
import edparser.components
import edparser.datasets
import edparser.layers
import edparser.losses
import edparser.metrics
import edparser.optimizers
import edparser.pretrained
import edparser.utils

from edparser.version import __version__

set_gpu_memory_growth()
#set_gpu()
ls_resource_in_module(edparser.pretrained)


def load(save_dir: str, meta_filename='meta.json', transform_only=False, load_kwargs=None,
         **kwargs) -> edparser.common.component.Component:
    """
    Load saved component from identifier.
    :param save_dir: The identifier to the saved component.
    :param meta_filename: The meta file of that saved component, which stores the class_path and version.
    :param transform_only: Whether to load transform only.
    :param load_kwargs: The arguments passed to `load`
    :param kwargs: Additional arguments parsed to the `from_meta` method.
    :return: A pretrained component.
    """
    save_dir = edparser.pretrained.ALL.get(save_dir, save_dir)
    from edparser.utils.component_util import load_from_meta_file
    return load_from_meta_file(save_dir, meta_filename, transform_only=transform_only, load_kwargs=load_kwargs,
                               **kwargs)


def pipeline(*pipes) -> edparser.components.pipeline.Pipeline:
    """
    Creates a pipeline of components.
    :param pipes: Components if pre-defined any.
    :return: A pipeline
    """
    return edparser.components.pipeline.Pipeline(*pipes)
