import cairo
import pango
import pangocairo
import rsvg
import spacy
import hashlib
import threading
from spacy.tokens import Span
from whoosh import index, scoring
from whoosh.fields import *
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import QueryParser
import os
import Levenshtein
import Image, ImageFilter, ImageColor, ImageMath, ImageOps, ImageDraw
from nltk.corpus import wordnet


MAX_ICONS = 12
MARGIN = 0.16
TOP_EXTRA_MARGIN = 0.05
ICON_SPACING = 0.04
FONT_FACE = 'sans'
DARK_ICONS = 1
OVERSAMPLING = 1.0
COLORIZE = 1

BACKGROUNDS = {'folder': 'folder3.svg', 'file': 'file3.svg',
               'collection': 'collection3.svg', 'text': 'text3.svg'}

RESOURCES_DIR = os.path.realpath('%s/../../resources' % os.path.dirname(os.path.realpath(__file__)))


class Layout:
    def __init__(self, nRows, nCols, rowColCount, rowIconSize=[1.0, 1.0, 1.0]):
        self.nRows = nRows
        self.nCols = nCols
        self.rowColCount = rowColCount
        self.rowIconSize = rowIconSize
        self.computeOffsets()

    def computeOffsets(self):
        self.yOffsets = []
        self.xOffsets = []

        h = 0.0
        w = 0.0
        rws = []
        for row in range(self.nRows):
            h = h + self.rowIconSize[row]
            rw = self.rowColCount[row] * self.rowIconSize[row]
            rws.append(rw)
            if rw > w:
                w = rw

        self.scale = 1.0 / max(w, h)

        rowMargins = []
        for row in range(self.nRows):
            rowMargins.append(self.scale * (w - rws[row]) / 2)

        dy = max(0.0, self.scale * (w - h) / 2)

        y = ICON_SPACING / 2 + dy
        for row in range(self.nRows):
            self.yOffsets.append(y)
            xColOffsets = []
            self.xOffsets.append(xColOffsets)

            x = rowMargins[row] + ICON_SPACING / 2
            for i in range(self.rowColCount[row]):
                xColOffsets.append(x)
                x = x + self.scale * self.rowIconSize[row]
            y = y + self.scale * self.rowIconSize[row]

        #print('layout: %s\n\ticonSize: %s\n\txOffsets: %s\n\tyOffsets: %s' %
        #      (self.rowColCount, [self.scale * x - ICON_SPACING for x in self.rowIconSize],
        #       self.xOffsets, self.yOffsets))

    def iconSize(self, row, scale=1.0):
        self.rowIconSize[row] * self.scale * scale

    def cellIter(self):
        for row in range(self.nRows):
            yOffset = self.yOffsets[row]
            for col in range(self.rowColCount[row]):
                xOffset = self.xOffsets[row][col]
                yield (xOffset, yOffset, self.scale * self.rowIconSize[row] - ICON_SPACING)


LAYOUTS = [None, Layout(1, 1, [1]), Layout(1, 2, [2]), Layout(2, 2, [1, 2]), Layout(2, 2, [2, 2]),
           Layout(2, 3, [2, 3], [1.5, 1.0]), Layout(2, 3, [3, 3]), Layout(3, 3, [1, 3, 3]),
           Layout(3, 3, [2, 3, 3]), Layout(3, 3, [3, 3, 3]), Layout(3, 4, [2, 4, 4]),
           Layout(3, 4, [3, 4, 4]), Layout(3, 4, [4, 4, 4])]

ENTS = set([u'ORG', u'GPE', u'LOC', u'PRODUCT', u'LANGUAGE', u'DATE', u'NORP'])
NO_SEM_SEARCH = set([u'GPE', u'NORP', u'LANGUAGE', u'DATE', u'LOC'])
SKIP_TOKENS = set([u'ADP', u'CCONJ', u'DET', u'PUNCT', u'SYM'])
IMAGE_TYPES = set(['.png', '.svg'])

class SearchSchema(SchemaClass):
    path = ID(stored=True)
    name = TEXT(analyzer=StemmingAnalyzer())

def getTargetColor(hash):
    hue = hash[0] * 359 / 255
    (r, g, b) = ImageColor.getrgb('hsl(%s, 30%%, 20%%)' % hue)
    return (r / 255.0, g / 255.0, b / 255.0)

class Special:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return 'Special[%s]' % self.value

    def paint(self, ctx, x, y, w, h, alpha, hash):
        ctx.select_font_face(FONT_FACE, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(32.0)
        xf, yf, wf, hf = ctx.text_extents(self.value)[:4]
        xscale = w / wf
        yscale = h / hf
        scale = min(xscale, yscale) * 0.85
        xflush = (w - scale * wf) / 2
        yflush = (h - scale * hf) / 2
        ctx.set_font_size(32.0 * scale)
        ctx.move_to(x + xflush, y + yflush + 24.0 * scale)
        if COLORIZE:
            r, g, b = getTargetColor(hash)
            ctx.set_source_rgba(r, g, b, alpha)
        elif DARK_ICONS:
            ctx.set_source_rgba(0.0, 0.0, 0.0, alpha)
        else:
            ctx.set_source_rgba(1.0, 1.0, 1.0, alpha)
        ctx.show_text(self.value)

class IconGenerator:
    def __init__(self, indexPath):
        self.iconPath = RESOURCES_DIR + '/icons'
        self.indexPath = indexPath
        self.nlp = spacy.load('en')
        self.iconVectors = {}
        self.iconPaths = {}
        self.initIndex()
        self.infoCache = {}
        self.infoLock = threading.Lock()

    def initIndex(self):
        if not os.path.exists(self.indexPath):
            os.makedirs(self.indexPath)
        self.index = index.create_in(self.indexPath, SearchSchema)
        self.updateIndex()

    def updateIndex(self):
        wr = self.index.writer()
        priorities = {}
        prefixLen = len(RESOURCES_DIR) + 1
        for dir, subdirs, files in os.walk(self.iconPath):
            priority = self._getIconDirPriority(dir)
            for file in files:
                ps = os.path.splitext(file)
                name = ps[0].lower().replace('_', ' ')
                ext = ps[1].lower()
                if not ext in IMAGE_TYPES:
                    continue
                uname = u'%s' % name
                if uname in priorities and priorities[uname] < priority:
                    continue
                path = u'%s/%s' % (dir, file)
                path = path[prefixLen:]
                priorities[uname] = priority
                pt = self.nlp(uname)
                self.iconVectors[uname] = pt
                self.iconPaths[uname] = path
                wr.add_document(path=path, name=uname)
        wr.commit()

    def _getIconDirPriority(self, dir):
        pfile = '%s/.priority' % dir
        if os.path.exists(pfile):
            with open(pfile) as p:
                try:
                    return int(p.readline().strip())
                except:
                    pass
        return 99

    def generateIcon(self, type, text, w, h, path, params):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
        ctx = cairo.Context(surface)
        hash = self.getHash(text, 2)
        self.drawBackground(ctx, text, type, w, h, hash)

        icons = self.getIcons(text)
        #icons = [u'/home/mike/work/wt/repos/wt_sils/resources/icons/water.svg', u'/home/mike/work/wt/repos/wt_sils/resources/icons/sample.svg', u'/home/mike/work/wt/repos/wt_sils/resources/icons/river.svg', u'/home/mike/work/wt/repos/wt_sils/resources/icons/geography/canada.svg', Special('2016')]
        print icons

        if len(icons) < 2:
            self.paintText(ctx, text, w, h, hash)
        else:
            self.paintIcons(ctx, icons, text, w, h, hash)

        im = self.surfaceToImage(surface)
        im = self.toGrayscale(im)
        shadowDepth = 0.01
        shadow = self.gaussianBlur(im, 1.5 * shadowDepth * w * OVERSAMPLING)
        surfs = self.imageToSurface(shadow)
        self.paintSurface(ctx, surfs, shadowDepth, shadowDepth, w, h, cairo.OPERATOR_DEST_OVER, 0.4)

        surface.write_to_png(path)

    def toGrayscale(self, im):
        # gotta keep the alpha channel
        r, g, b, a = im.split()
        im = im.convert('L').convert('RGB')
        return Image.merge('RGBA', (r, g, b, a))

    def gaussianBlur(self, im, radius):
        return im.filter(ImageFilter.GaussianBlur(radius=radius))

    def paintText(self, ctx, text, w, h, hash):
        ctx.save()
        mx = int(w * MARGIN)
        my = int(h * MARGIN)
        mw = int(w * (1 - 2 * MARGIN))
        mh = int(w * (1 - 2 * MARGIN - TOP_EXTRA_MARGIN))
        if COLORIZE:
            r, g, b = getTargetColor(hash)
            print('colorize: %s, %s, %s' % (r, g, b))
            ctx.set_source_rgb(r, g, b)
        elif DARK_ICONS:
            ctx.set_source_rgb(0.0, 0.0, 0.0)
        else:
            ctx.set_source_rgb(1.0, 1.0, 1.0)
        pgc = pangocairo.CairoContext(ctx)
        pgc.set_antialias(cairo.ANTIALIAS_SUBPIXEL)

        layout = pgc.create_layout()
        fontname = FONT_FACE
        fontsz = int(h / 20)
        font = pango.FontDescription('%s %s' % (fontname, fontsz))
        layout.set_font_description(font)

        layout.set_wrap(pango.WRAP_WORD)
        layout.set_width(int(pango.SCALE * w * (1 - 2 * MARGIN)))
        layout.set_text(text)
        pgc.rectangle(mx, my, mw, mh)
        pgc.clip()

        (tw, th) = layout.get_pixel_size()
        if th < mh / 2:
            font = pango.FontDescription('%s %s' % (fontname, fontsz * 1.3))
            layout.set_font_description(font)
            (tw, th) = layout.get_pixel_size()
        dy = max(0, (mh - th) / 2)
        pgc.move_to(mx, my + dy)

        pgc.update_layout(layout)
        pgc.show_layout(layout)
        ctx.restore()

    def paintIcons(self, ctx2, icons, text, w, h, hash):
        if len(icons) == 0:
            return
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
        ctx = cairo.Context(surface)
        if len(icons) > MAX_ICONS:
            icons = icons[0:MAX_ICONS]

        scale = (1.0 - 2 * MARGIN)
        layout = LAYOUTS[len(icons)]
        crd = layout.cellIter()
        for icon in icons:
            (x, y, iconSize) = crd.next()
            px = MARGIN + x * scale
            py = MARGIN + y * scale + TOP_EXTRA_MARGIN
            iconSize = iconSize * scale
            ctx.save()
            self.paintIcon(ctx, icon, w, h, px, py, iconSize, hash)
            ctx.restore()

        ctx2.set_source_surface(surface)
        ctx2.paint()

    def getHash(self, text, nBytes):
        digest = hashlib.sha224(text).digest()
        return [ord(x) for x in digest[:nBytes]]

    def _scale(self, v, l, h):
        return l + (h - l) * (float(v) / 255)

    def paintIcon(self, ctx, icon, w, h, x, y, iconSize, hash):
        if isinstance(icon, Special):
            icon.paint(ctx, x * w, y * h, iconSize * w, iconSize * h, 0.99, hash)
        else:
            iconW = int(iconSize * w)
            iconH = iconW
            surf = self._loadImage(icon, iconW, iconH)

            im1 = self.surfaceToImage(surf)

            print('Image: %s' % icon)
            (grayscale, mostlyDark, heavy) = self.getImageProperties(im1, icon)

            print('%s - grayscale: %s, mostlyDark: %s, heavy: %s' % (icon, grayscale, mostlyDark, heavy))


            if grayscale:
                #if heavy:
                #    im1 = self.edgeDetect(im1)
                if mostlyDark and not DARK_ICONS or DARK_ICONS and not mostlyDark:
                    im1 = self.invert(im1)

                im1 = self.colorize(im1, hash)
                surf = self.imageToSurface(im1)

                self.paintSurface(ctx, surf, x, y, w, h, cairo.OPERATOR_ADD + 4, 0.99)
            else:
                self.paintSurface(ctx, surf, x, y, w, h, alpha=0.8)

    def paintSurface(self, ctx, surf, x, y, w, h, op=None, alpha=1.0):
        ctx.save()
        ctx.translate(x, y)
        ctx.scale(1.0 / OVERSAMPLING, 1.0 / OVERSAMPLING)
        ctx.set_source_surface(surf, x * w * OVERSAMPLING, y * h * OVERSAMPLING)
        if op:
            ctx.set_operator(op)
        ctx.paint_with_alpha(alpha)
        ctx.restore()

    def surfaceToImage(self, surf):
        return Image.frombuffer("RGBA", (surf.get_width(), surf.get_height()),
                               surf.get_data(), 'raw', 'RGBA', 0, 1)

    def imageToSurface(self, im):
        if 'A' not in im.getbands():
            im.putalpha(256)
        arr = bytearray(im.tobytes('raw', 'BGRa'))
        return cairo.ImageSurface.create_for_data(arr, cairo.FORMAT_ARGB32, im.width,
                                                  im.height)

    def colorize(self, im, hash):
        (cr, cg, cb) = getTargetColor(hash)
        (cr, cg, cb) = (int(255 * cr), int (255 * cg), int(255 * cb))
        r, g, b, alpha = im.split()
        gray = ImageOps.grayscale(im)
        result = ImageOps.colorize(gray, (cr, cg, cb), (255, 255, 255))
        result.putalpha(alpha)
        return result

    def edgeDetect(self, im1):
        return im1.filter(ImageFilter.FIND_EDGES)

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

    def getImageProperties(self, im, path):
        with self.infoLock:
            if path in self.infoCache:
                return self.infoCache[path]

            r, g, b, a = im.split()

            im2 = im.convert('HSV')
            (h, avol) = self._weightedHistogram(im2, a)
            vol = float(im2.width * im2.height)

            sa = h[256:512]
            va = h[512:768]

            smoment = self._moment(sa) / avol
            grayscale = (smoment < 10)

            vmoment = self._moment(va) / avol
            mostlyDark = (vmoment < 128)

            amoment = self._moment(a.histogram()) / vol
            heavy = (amoment > 90)

            print('smoment: %s, vmoment: %s, amoment: %s' % (smoment, vmoment, amoment))

            result = (grayscale, mostlyDark, heavy)
            self.infoCache[path] = result

            return result

    def _weightedHistogram(self, im, a):
        h = [0.0 for i in range(768)]
        imd = im.getdata()
        ad = a.getdata()
        avol = 0.0

        for i in range(len(imd)):
            pxa = ad[i]
            if pxa == 0:
                continue
            px = imd[i]
            w = pxa / 255.0
            spx = px[1]
            vpx = px[2]
            # saturation isn't quite enough to tell if an image is grayscale
            # since black/white can be represented as (h=x, s=y, v=0.0), (h=x, s=y, v=1.0)
            # (i.e., for any value of h and s)
            aspx = int(2 * spx * ((128 + abs(vpx - 128)) / 255.0))
            h[aspx + 256] += w
            h[vpx + 512] += w
            avol += w

        return (h, avol)

    def _moment(self, h):
        m = 0.0
        for i in range(256):
            m = m + h[i] * i
        return m

    def getIcons(self, text):

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

        print lst


        lst2 = []
        qp = QueryParser('name', schema=self.index.schema)
        with self.index.searcher() as s:
            for item in lst:
                r = self.searchWhoosh(qp, s, item)
                if r is not None:
                    # we have a hit
                    lst2.append(r[0]['path'])
                    print('QS %s -> %s (score: %s)' % (item.lemma_, r[0], r.score(0)))
                elif isinstance(item, Span):
                    # use the words
                    for token in item:
                        if not self.shouldAddToken(token):
                            print('SKIP %s (%s)' % (token.lemma_, token.pos_))
                            continue
                        r = self.searchWhoosh(qp, s, token)
                        if r is not None:
                            print('QT %s -> %s (score: %s)' % (token.lemma_, r[0], r.score(0)))
                            lst2.append(r[0]['path'])
                        else:
                            r = self.searchEntity(token)
                            if r is not None:
                                print('EN %s -> %s' % (token.lemma_, r))
                                lst2.append(r)
                            else:
                                r = None
                                if token.ent_type_ not in NO_SEM_SEARCH:
                                    # don't do a semantic search; we don't want to get
                                    # "american samoa" for "american"
                                    r = self.searchSemantic(token)
                                if r is not None:
                                    lst2.append(r)
                                    print('SE %s -> %s' % (token.lemma_, r))
                                else:
                                    r = self.makeSpecials(token, pt)
                                    if r is not None:
                                        lst2.append(r)
                                        print('ES %s -> %s' % (token.lemma_, r))
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
                                print('SE %s -> %s' % (item.lemma_, r))
                            else:
                                # leave it out, unfortunately
                                print('xxxx %s' % item.lemma_)
        return lst2

    def searchWhoosh(self, qp, searcher, token):
        q = self.buildQuery(qp, token)
        r = searcher.search(q)
        if len(r) > 0 and r.score(0) > 7:
            return r
        else:
            return None

    def searchEntity(self, token):
        if token.ent_iob != 2:
            if token.ent_type_ == u'DATE':
                return Special(token.lemma_)
            else:
                r = self.searchRelated(token)
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
                print('Similarity(%s, %s) = %s' % (item.lemma_, k, sim))
                results.append([sim, k])
        if len(results) == 0:
            return None
        results = sorted(results, key=lambda x: x[0], reverse=True)
        return self.iconPaths[results[0][1]]

    def searchSimilar(self, item):
        #
        # probably not very useful, since it can easily confuse "arctic" with "attic"
        #
        results = []
        for k in self.iconPaths:
            sim = float(Levenshtein.distance(item.lemma_, k)) / len(item.lemma_)
            if sim < 0.4:
                results.append([sim, k])
        if len(results) == 0:
            return None
        results = sorted(results, key=lambda x: x[0])
        return self.iconPaths[results[0][1]]

    def searchRelated(self, item):
        nouns = self.nounify(item.lemma_)
        print('nouns(%s) = %s' % (item.lemma_, nouns))
        for noun in nouns:
            if noun in self.iconPaths:
                return self.iconPaths[noun]
        return None

    def nounify(self, adj):
        related = set()

        morphy = wordnet.morphy(adj, wordnet.NOUN)
        if morphy:
            for lemma in wordnet.lemmas(morphy):
                for rf in lemma.derivationally_related_forms():
                    for synset in wordnet.synsets(rf.name(), pos=wordnet.NOUN):
                        related.add(synset.name().split('.')[0].lower())

        return related

    def shouldAddToken(self, token):
        if token.pos_ in SKIP_TOKENS:
            return False
        if token.pos_ == u'NUM' and token.ent_type_ == u'CARDINAL':
            return False
        return True


    def drawBackground(self, ctx, text, type, w, h, hash):
        bgName = self.getResource(['backgrounds', BACKGROUNDS[type]])
        surface = self._loadSvg(bgName, w, h)
        surface = self.modifyBackground(surface, text, w, h, hash)
        self.paintSurface(ctx, surface, 0, 0, w, h)

    def modifyBackground(self, surface, text, w, h, hash):
        im = self.surfaceToImage(surface)
        im = self.modifyColors(im, w, h, text, hash)
        return self.imageToSurface(im)

    def modifyColors(self, im, w, h, text, hash):
        hd = hash[0]
        vd = int(self._scale(hash[1], -0.2, 0.2) * 255)

        r, g, b, a = im.split()
        imrgb = Image.merge('RGB', (r, g, b))
        imhsv = imrgb.convert('HSV')

        h, s, v = imhsv.split()

        h = ImageMath.eval("""convert((h + hd) % 256, 'L')""", h=h, hd=hd)
        v = ImageMath.eval("""convert(min(max(v + vd, 0), 255), 'L')""", v=v, vd=vd)
        print('%s, %s, %s' % (h.mode, s.mode, v.mode))
        imhsv = Image.merge('HSV', (h, s, v))
        imrgb = imhsv.convert('RGB')
        imrgb.putalpha(a)
        return imrgb

    def _loadImage(self, name, w, h):
        name = RESOURCES_DIR + '/' + name
        if name.endswith('.svg'):
            return self._loadSvg(name, w, h)
        elif name.endswith('.png'):
            return self._loadPng(name, w, h)
        else:
            print('Unknown image type: %s' % name)
            return self._emptyImage(name, w, h)

    def _emptyImage(self, w, h):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
        ctx = cairo.Context(surface)
        ctx.set_source_rgba(1.0, 0.0, 0.0, 0.0)
        ctx.move_to(0, 0)
        ctx.line_to(w, h)
        ctx.move_to(w, 0)
        ctx.line_to(0, h)
        ctx.paint()
        return surface

    def _loadPng(self, name, w, h):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,
                                     int(w * OVERSAMPLING), int(h * OVERSAMPLING))
        ctx = cairo.Context(surface)
        background = cairo.ImageSurface.create_from_png(name)
        ctx.scale(float(w * OVERSAMPLING) / background.get_width(),
                  float(h * OVERSAMPLING) / background.get_height())
        ctx.set_source_surface(background)
        ctx.paint()
        return surface

    def _loadSvg(self, name, w, h):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,
                                     int(w * OVERSAMPLING), int(h * OVERSAMPLING))
        ctx = cairo.Context(surface)
        background = rsvg.Handle(file=name)
        (bw, bh, bw2, bh2) = background.get_dimension_data()
        ctx.scale(float(w * OVERSAMPLING) / bw,
                  float(h * OVERSAMPLING) / bh)
        background.render_cairo(ctx)
        return surface


    def getResource(self, plist):
        plist.insert(0, RESOURCES_DIR)
        return '/'.join(plist)

if __name__ == "__main__":
    ig = IconGenerator('/tmp/sils-cache')
    ig.generateIcon('folder', 'Water samples from rivers and estuaries in the Canadian Arctic Archipelago, 2016', 256, 256, '/tmp/col/img1.png', {})
    ig.generateIcon('folder', 'Togiak Archaeological and Paleoecological Project', 256, 256, '/tmp/col/img2.png', {})
    ig.generateIcon('folder', 'Spatiotemporal Gypsy Moth Defoliation Data, Northeastern USA, 1975-2009', 256, 256, '/tmp/col/img3.png', {})
    ig.generateIcon('folder', 'Photogrammetric scans of aerial photographs of North American glaciers, 1992. Roll 5 tiffs', 256, 256, '/tmp/col/img4.png', {})
    ig.generateIcon('folder', 'SNAPP - Mapping the Global Potential for Marine Aquaculture', 256, 256, '/tmp/col/img5.png', {})
    ig.generateIcon('folder', 'Humans and Hydrology at High Latitudes: Water Use Information', 256, 256, '/tmp/col/img6.png', {})

