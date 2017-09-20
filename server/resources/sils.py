import hashlib

try:
    from gi.repository import Rsvg as rsvg
except ImportError:
    import rsvg

from girder.api import rest
from girder.api.rest import Resource, RestException
from girder.api.rest import loadmodel
from girder.constants import AccessType
from girder.api import access
from girder.api.describe import Description, describeRoute
from ..lib.icongenerator2 import IconGenerator
import threading
import os


def _getParam(params, name, default=None):
    if name in params:
        return params[name]
    else:
        if default is None:
            raise RestException('Missing parameter %s' % name)
        else:
            return default

def _getIntParam(params, name, default=None):
    if name in params:
        try:
            return int(params[name])
        except:
            raise RestException('Invalid value for parameter %s: %s. Expected an integer' %
                                (name, params[name]))
    else:
        if default is None:
            raise RestException('Missing parameter %s' % name)
        else:
            return default

class SILS(Resource):
    def __init__(self, cachePath):
        Resource.__init__(self)
        self.cachePath = cachePath
        self.iconGenerator = IconGenerator(cachePath + '/index')
        self.cacheLock = threading.Lock()
        self.generating = {}


    @access.public
    @loadmodel(model='folder', level=AccessType.READ)
    @describeRoute(
        Description('Generate an icon for a folder.')
            .param('id', 'The ID of the folder.', paramType='path')
            .param('w', 'The width of the icon. Default is 100.', paramType='query')
            .param('h', 'The height of the icon. Default is 100.', paramType='query')
            .errorResponse('ID was invalid.')
            .errorResponse('Read access was denied for the folder.', 403)
    )
    def folderIcon(self, folder, params):
        return self.icon('folder', self._getName(params['id']), params)

    @access.public
    @loadmodel(model='collection', level=AccessType.READ)
    @describeRoute(
        Description('Generate an icon for a collection.')
            .param('id', 'The ID of the collection.', paramType='path')
            .param('w', 'The width of the icon. Default is 100.', paramType='query')
            .param('h', 'The height of the icon. Default is 100.', paramType='query')
            .errorResponse('ID was invalid.')
            .errorResponse('Read access was denied for the collection.', 403)
    )
    def collectionIcon(self, collection, params):
        return self.icon('collection', self._getName(params['id']), params)

    @access.public
    @loadmodel(model='item', level=AccessType.READ)
    @describeRoute(
        Description('Generate an icon for an item.')
            .param('id', 'The ID of the item.', paramType='path')
            .param('w', 'The width of the icon. Default is 100.', paramType='query')
            .param('h', 'The height of the icon. Default is 100.', paramType='query')
            .errorResponse('ID was invalid.')
            .errorResponse('Read access was denied for the item.', 403)
    )
    def itemIcon(self, item, params):
        return self.icon('item', self._getName(params['id']), params)

    @access.public
    @describeRoute(
        Description('Generate an icon from a text snippet.')
            .param('text', 'The text to generate the icon from.', paramType='query')
            .param('w', 'The width of the icon. Default is 100.', paramType='query')
            .param('h', 'The height of the icon. Default is 100.', paramType='query')
    )
    def iconFromText(self, params):
        return self.icon('text', _getParam(params, 'text'), params)

    def icon(self, type, text, params):
        w = _getIntParam(params, 'w', 160)
        h = _getIntParam(params, 'h', 160)

        hash = hashlib.sha224('%s-%s-%s-%s' % (type, text, w, h)).hexdigest()
        path = self.cachePath + '/' + hash

        otherLock = None
        myLock = None
        with self.cacheLock:
            if hash in self.generating:
                otherLock = self.generating[hash]
            else:
                if os.path.isfile(path):
                    # release cache lock and serve cached file
                    pass
                else:
                    # add imageLock and generate
                    myLock = threading.Lock()
                    self.generating[hash] = myLock
                    myLock.acquire()

        if otherLock is not None:
            otherLock.acquire()
            # done generating
            otherLock.release()
        elif myLock is not None:
            self.iconGenerator.generateIcon(type, text, w, h, path, params)
            with self.cacheLock:
                del self.generating[hash]
                myLock.release()
        else:
            # neither this thread or another thread is working on this
            pass

        return self._serveFile(path)

    def _serveFile(self, path):
        rest.setResponseHeader('Content-Type', 'image/png')
        def gen():
            f = open(path)
            for line in f:
                yield line
            f.close()
        return gen