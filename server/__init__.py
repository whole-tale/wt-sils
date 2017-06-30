#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .constants import PluginSettings
from .resources.sils import SILS
from girder.models.setting import Setting
from girder.utility import setting_utilities
from girder.constants import SettingDefault


@setting_utilities.validator({
    PluginSettings.ICON_CACHE_PATH,
    PluginSettings.RESOURCES_PATH
})
def validateOtherSettings(event):
    pass


def load(info):
    SettingDefault.defaults[PluginSettings.ICON_CACHE_PATH] = '/tmp/sils-cache'
    SettingDefault.defaults[PluginSettings.RESOURCES_PATH] = '/home/mike/work/wt/repos/wt_sils/resources'

    settings = Setting()

    sils = SILS(settings.get(PluginSettings.RESOURCES_PATH))

    info['apiRoot'].sils = sils

    info['apiRoot'].folder.route('GET', (':id', 'icon'), sils.folderIcon)
    info['apiRoot'].collection.route('GET', (':id', 'icon'), sils.collectionIcon)
    info['apiRoot'].item.route('GET', (':id', 'icon'), sils.itemIcon)
    info['apiRoot'].sils.route('GET', (), sils.iconFromText)
