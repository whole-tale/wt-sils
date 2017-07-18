import cairo

try:
    from gi.repository import Rsvg as rsvg
except ImportError:
    import rsvg

from girder.api.rest import Resource
from girder.api.rest import loadmodel
from girder.constants import AccessType
from girder.api import access
from girder.api.describe import Description, describeRoute


def _getParam(params, name, default):
    if name in params:
        return params[name]
    else:
        return default

def _getIntParam(params, name, default):
    if name in params:
        try:
            return int(params[name])
        except:
            return default
    else:
        return default

class SILS(Resource):
    def __init__(self, resourcesPath):
        Resource.__init__(self)
        self.resourcesPath = resourcesPath
        pass

    @access.user
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
        return self.icon('folder', params)

    @access.user
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
        return self.icon('collection', params)

    @access.user
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
        return self.icon('item', params)

    @access.user
    @describeRoute(
        Description('Generate an icon from a text snippet.')
            .param('text', 'The text to generate the icon from.', paramType='query')
            .param('w', 'The width of the icon. Default is 100.', paramType='query')
            .param('h', 'The height of the icon. Default is 100.', paramType='query')
    )
    def iconFromText(self, params):
        return self.icon('text', params)

    def icon(self, type, params):
        w = _getIntParam(params, 'w', 100)
        h = _getIntParam(params, 'h', 160)

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
        ctx = cairo.Context(surface)
        ctx.scale(w, h)

        background = rsvg.Handle(file=self.getResource(['backgrounds', type]))
        background.render(ctx)



    def getResource(self, plist):
        plist.insert(0, self.resourcesPath)
        return '/'.join(plist)
