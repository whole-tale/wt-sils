#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .constants import PluginSettings
from .resources.sils import SILS
from girder.models.setting import Setting
from girder.utility import setting_utilities
from girder.constants import SettingDefault
import os


@setting_utilities.validator({
    PluginSettings.ICON_CACHE_PATH,
})
def validateOtherSettings(event):
    pass


def load(info):
    SettingDefault.defaults[PluginSettings.ICON_CACHE_PATH] = '/tmp/sils-cache'

    settings = Setting()

    cachePath = settings.get(PluginSettings.ICON_CACHE_PATH)

    if not os.path.exists(cachePath):
        os.makedirs(cachePath)

    sils = SILS(cachePath)

    info['apiRoot'].sils = sils

    info['apiRoot'].folder.route('GET', (':id', 'icon'), sils.folderIcon)
    info['apiRoot'].collection.route('GET', (':id', 'icon'), sils.collectionIcon)
    info['apiRoot'].item.route('GET', (':id', 'icon'), sils.itemIcon)
    info['apiRoot'].sils.route('GET', (), sils.iconFromText)
