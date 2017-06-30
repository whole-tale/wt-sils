import router from 'girder/router';
import events from 'girder/events';
import { exposePluginConfig } from 'girder/utilities/PluginUtils';

exposePluginConfig('wt_sils', 'plugins/wt_sils/config');

import ConfigView from './views/ConfigView';
router.route('plugins/wt_sils/config', 'SILSConfig', function () {
    events.trigger('g:navigateTo', ConfigView);
});
