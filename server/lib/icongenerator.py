import cairo
import pango
import pangocairo
import rsvg
import spacy
import hashlib
import colorsys
from spacy.tokens import Span
from whoosh import index, scoring
from whoosh.fields import *
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import QueryParser
import os
import Levenshtein
import Image, ImageFilter, ImageChops, ImageMath, ImageOps

def distance2(a, b):
    return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2])

class Layout:
    def __init__(self, nRows, nCols, rowColCount):
        self.nRows = nRows
        self.nCols = nCols
        self.rowColCount = rowColCount

    def iconSize(self):
        maxIconCells = max(self.nCols, self.nRows)
        return (1.0 - 2 * ICON_FLUSH - (maxIconCells - 1) * ICON_SPACING) / maxIconCells

    def cellIter(self):
        for row in range(self.nRows):
            yOffset = self._yOffset()
            for col in range(self.rowColCount[row]):
                yield (col + self._xOffset(row), row + yOffset)

    def _yOffset(self):
        return (max(self.nCols, self.nRows) - self.nRows) / 2.0

    def _xOffset(self, row):
        return (max(self.nCols, self.nRows) - self.rowColCount[row]) / 2.0


LAYOUTS = [None, Layout(1, 1, [1]), Layout(1, 2, [2]), Layout(2, 2, [1, 2]), Layout(2, 2, [2, 2]),
           Layout(2, 3, [3, 2]), Layout(2, 3, [3, 3]), Layout(3, 3, [3, 3, 1]),
           Layout(3, 3, [3, 3, 2]), Layout(3, 3, [3, 3, 3]), Layout(3, 4, [4, 4, 2]),
           Layout(3, 4, [4, 4, 3]), Layout(3, 4, [4, 4, 4])]

MAX_ICONS = 12
ICON_FLUSH = 0.2
TOP_SHIFT = 0.05
ICON_SPACING = 0.02
FONT_FACE = 'sans'

ENTS = set([u'ORG', u'GPE', u'LOC', u'PRODUCT', u'LANGUAGE', u'DATE'])
SKIP_TOKENS = set([u'ADP', u'CCONJ', u'DET', u'PUNCT', u'SYM'])

class SearchSchema(SchemaClass):
    path = ID(stored=True)
    name = TEXT(analyzer=StemmingAnalyzer())

class Special:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return 'Special[%s]' % self.value

    def paint(self, ctx, x, y, w, h, alpha):
        ctx.set_source_rgba(0, 0, 0, alpha)
        ctx.select_font_face(FONT_FACE, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(32.0)
        xf, yf, wf, hf = ctx.text_extents(self.value)[:4]
        xscale = w / wf
        yscale = h / hf
        scale = min(xscale, yscale) * 0.8
        xflush = (w - scale * wf) / 2
        yflush = (h - scale * hf) / 2
        print("scale: %s, x: %s, y: %s" % (scale, x, y))
        ctx.set_font_size(32.0 * scale)
        ctx.move_to(x + xflush, y + yflush + 16.0 * scale)
        ctx.show_text(self.value)

class IconGenerator:
    def __init__(self, resourcesPath, indexPath):
        self.resourcesPath = resourcesPath
        self.iconPath = self.resourcesPath + '/icons'
        self.indexPath = indexPath
        self.nlp = spacy.load('en')
        self.iconVectors = {}
        self.iconPaths = {}
        self.initIndex()

    def initIndex(self):
        if not os.path.exists(self.indexPath):
            os.makedirs(self.indexPath)
        self.index = index.create_in(self.indexPath, SearchSchema)
        self.updateIndex()

    def updateIndex(self):
        wr = self.index.writer()
        for dir, subdirs, files in os.walk(self.iconPath):
            for file in files:
                name = os.path.splitext(file)[0].lower().replace('_', ' ')
                path = u'%s/%s' % (dir, file)
                uname = u'%s' % name
                pt = self.nlp(uname)
                self.iconVectors[uname] = pt
                self.iconPaths[uname] = path
                wr.add_document(path=path, name=uname)
        wr.commit()

    def generateIcon(self, type, text, w, h, path, params):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
        ctx = cairo.Context(surface)
        ctx.save()
        self.drawBackground(ctx, type, w, h, 0.9)
        ctx.set_operator(cairo.OPERATOR_ATOP)
        self.setGradient(ctx, text, w, h)
        #ctx.set_operator(cairo.OPERATOR_OVER)
        #self.drawBackground(ctx, type, w, h, 0.1)
        #self.drawOverlay(ctx, w, h)
        ctx.restore()

        icons = self.getIcons(text)
        #icons = [u'/home/mike/work/wt/repos/wt_sils/resources/icons/water.svg', u'/home/mike/work/wt/repos/wt_sils/resources/icons/sample.svg', u'/home/mike/work/wt/repos/wt_sils/resources/icons/river.svg', u'/home/mike/work/wt/repos/wt_sils/resources/icons/geography/canada.svg', Special('2016')]
        print icons

        self.paintIcons(ctx, icons, text, w, h)

        surface.write_to_png(path)

    def paintIcons(self, ctx2, icons, text, w, h):
        if len(icons) == 0:
            return
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
        ctx = cairo.Context(surface)
        if len(icons) > MAX_ICONS:
            icons = icons[0:MAX_ICONS]

        layout = LAYOUTS[len(icons)]
        iconSize = layout.iconSize()
        crd = layout.cellIter()
        for icon in icons:
            (x, y) = crd.next()
            px = ICON_FLUSH + x * (ICON_SPACING + iconSize)
            py = ICON_FLUSH + y * (ICON_SPACING + iconSize) + TOP_SHIFT
            ctx.save()
            self.paintIcon(ctx, icon, w, h, px, py, iconSize)
            ctx.restore()

        ctx2.set_source_surface(surface)
        ctx2.paint()

    def _scale(self, v, l, h):
        return l + (h - l) * (float(v) / 255)

    def paintIcon(self, ctx, icon, w, h, x, y, iconSize):
        if isinstance(icon, Special):
            icon.paint(ctx, x * w, y * h, iconSize * w, iconSize * h, 0.7)
        else:
            iconW = int(iconSize * w)
            iconH = iconW
            surf = self.loadSvg(icon, iconW, iconH)

            im1 = Image.frombuffer("RGBA", (surf.get_width(), surf.get_height()),
                                       surf.get_data(), 'raw', 'RGBA', 0, 1)
            if self.isGrayscale(im1):
                print('%s - grayscale' % icon)
                im1 = im1.filter(ImageFilter.FIND_EDGES)
                #im1 = self.colorToAlpha(im1, (255, 255, 255), 5)
                im1 = self.invert(im1)
                #im1 = self.emboss(im1)
                if 'A' not in im1.getbands():
                    im1.putalpha(256)
                arr = bytearray(im1.tobytes('raw', 'BGRa'))
                surf = cairo.ImageSurface.create_for_data(arr, cairo.FORMAT_ARGB32, im1.width,
                                                          im1.height)

                ctx.set_source_surface(surf, x * w, y * h)
                ctx.translate(x, y)
                ctx.set_operator(cairo.OPERATOR_ADD + 3)
                ctx.paint_with_alpha(0.8)
            else:
                print('%s - color' % icon)
                ctx.set_source_surface(surf, x * w, y * h)
                ctx.translate(x, y)
                ctx.paint_with_alpha(0.9)


    def colorToAlpha(self, image, color, thresh2=0):
        red, green, blue, alpha = image.split()
        image.putalpha(
            ImageMath.eval("""convert(((((t - d(c, (r, g, b))) >> 31) + 1) ^ 1) * a, 'L')""",
                           t=thresh2, d=distance2, c=color, r=red, g=green, b=blue, a=alpha))
        return image

    def invert(self, image):
        r, g, b, a = image.split()
        rgbi = Image.merge('RGB', (r, g, b))

        inverted = ImageOps.invert(rgbi)
        inverted.putalpha(a)

        return inverted

    def emboss(self, image):
        r, g, b, a = image.split()
        rgbi = Image.merge('RGB', (r, g, b))

        embossed = rgbi.filter(ImageFilter.EMBOSS)
        embossed.putalpha(a)

        return embossed

    def isGrayscale(self, im):
        im2 = im.convert('HSV')
        h = im2.histogram()
        vol = float(im2.width * im2.height)
        moment = 0.0
        for i in range(256, 512):
            moment = moment + h[i] * (i - 256) / vol
        return moment < 10

    def getIcons(self, text):
        # I'm not proud of this method

        pt = self.nlp(u'%s' % text)

        for token in pt:
            print(token.pos_, token.orth_, token.dep_, token.head.orth_, token.ent_iob, token.ent_type_, [t.orth_ for t in token.lefts],
                  [t.orth_ for t in token.rights])

        processedTokens = set()

        for chunk in pt.noun_chunks:
            for token in chunk:
                processedTokens.add(token)

        # prefer chunks to individual tokens
        # we revert to tokens later based on search results
        lst = []
        lastChunk = None
        lastIndex = 0
        for chunk in pt.noun_chunks:
            ixs = chunk.start
            for i in range(lastIndex, ixs):
                token = pt[i]
                if self.shouldAddToken(token):
                    lst.append(token)
            lst.append(chunk)
            lastChunk = chunk
            lastIndex = chunk.end + 1

        for i in range(lastIndex, len(pt)):
            token = pt[i]
            if self.shouldAddToken(token):
                lst.append(token)

        print('preliminary icons: %s' % lst)

        lst2 = []
        qp = QueryParser('name', schema=self.index.schema)
        with self.index.searcher() as s:
            for item in lst:
                q = self.buildQuery(qp, item)
                r = s.search(q)
                if len(r) > 0:
                    # we have a hit
                    lst2.append(r[0]['path'])
                    print('QS %s -> %s' % (item.lemma_, r[0]))
                elif isinstance(item, Span):
                    # use the words
                    for token in item:
                        if token.pos_ in SKIP_TOKENS:
                            print('SKIP %s (%s)' % (token.lemma_, token.pos_))
                            continue
                        q = self.buildQuery(qp, token)
                        r = s.search(q)
                        if len(r) > 0:
                            print('QT %s -> %s' % (token.lemma_, r[0]))
                            lst2.append(r[0]['path'])
                        else:
                            r = self.searchEntity(token)
                            if r is not None:
                                print('EN %s -> %s' % (token.lemma_, r))
                                lst2.append(r)
                            else:
                                r = self.searchSemantic(token)
                                if r is not None:
                                    lst2.append(r)
                                    print('SE %s -> %s' % (item.lemma_, r))
                                else:
                                    r = self.makeSpecials(token, pt)
                                    if r is not None:
                                        lst2.append(r)
                                        print('ES %s -> %s' % (item.lemma_, r))
                                    else:
                                        print('xxx %s' % token.lemma_)
                else:
                    if item.pos_ not in SKIP_TOKENS:
                        r = self.searchEntity(item)
                        if r is not None:
                            print('EN %s -> %s' % (item.lemma_, r))
                            lst2.append(r)
                        else:
                            r = self.searchSemantic(item)
                            if r is not None:
                                lst2.append(r)
                                print('%s -> %s' % (item.lemma_, r))
                            else:
                                # leave it out, unfortunately
                                print('xxxx %s' % item.lemma_)
        return lst2

    def searchEntity(self, token):
        if token.ent_iob != 2:
            if token.ent_type_ == u'DATE':
                return Special(token.lemma_)
            else:
                r = self.searchSimilar(token)
                if r is not None:
                    print('SI %s -> %s' % (token.lemma_, r))
                    return r
                else:
                    print('xxx %s' % token.lemma_)
                    return None

    def makeSpecials(self, token, pt):
        if token.ent_type_ == u'ORG' and not token.text in pt.vocab:
            return Special(token.text)

    def buildQuery(self, qp, item):
        q = qp.parse(item.lemma_)
        q.vector = item.vector
        return q

    def searchSemantic(self, item):
        results = []
        for k in self.iconVectors:
            sim = item.similarity(self.iconVectors[k])
            if sim > 0.7:
                results.append([sim, k])
        if len(results) == 0:
            return None
        results = sorted(results, key=lambda x: x[0], reverse=True)
        return self.iconPaths[results[0][1]]

    def searchSimilar(self, item):
        results = []
        for k in self.iconPaths:
            sim = float(Levenshtein.distance(item.lemma_, k)) / len(item.lemma_)
            if sim < 0.4:
                results.append([sim, k])
        if len(results) == 0:
            return None
        results = sorted(results, key=lambda x: x[0])
        return self.iconPaths[results[0][1]]

    def shouldAddToken(self, token):
        if token.pos_ in SKIP_TOKENS:
            return False
        return True

    def drawBackground(self, ctx, type, w, h, alpha):
        bgName = self.getResource(['backgrounds', type]) + '.svg'
        surface = self.loadSvg(bgName, w, h)
        ctx.set_source_surface(surface)
        ctx.paint_with_alpha(alpha)

    def loadSvg(self, name, w, h):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
        ctx = cairo.Context(surface)
        background = rsvg.Handle(file=name)
        (bw, bh, bw2, bh2) = background.get_dimension_data()
        ctx.scale(float(w) / bw, float(h) / bh)
        background.render_cairo(ctx)
        return surface

    def getResource(self, plist):
        plist.insert(0, self.resourcesPath)
        return '/'.join(plist)

if __name__ == "__main__":
    ig = IconGenerator('/home/mike/work/wt/repos/wt_sils/resources', '/tmp/sils-cache')
    ig.generateIcon('folder', 'Water samples from rivers and estuaries in the Canadian Arctic Archipelago, 2016', 256, 256, '/tmp/col/img1.png', {})
    ig.generateIcon('folder', 'Togiak Archaeological and Paleoecological Project', 256, 256, '/tmp/col/img2.png', {})
    ig.generateIcon('folder', 'Spatiotemporal Gypsy Moth Defoliation Data, Northeastern USA, 1975-2009', 256, 256, '/tmp/col/img3.png', {})
    ig.generateIcon('folder', 'Photogrammetric scans of aerial photographs of North American glaciers, 1992. Roll 5 tiffs', 256, 256, '/tmp/col/img4.png', {})
    ig.generateIcon('folder', 'SNAPP - Mapping the Global Potential for Marine Aquaculture', 256, 256, '/tmp/col/img5.png', {})
    ig.generateIcon('folder', 'Humans and Hydrology at High Latitudes: Water Use Information', 256, 256, '/tmp/col/img6.png', {})

