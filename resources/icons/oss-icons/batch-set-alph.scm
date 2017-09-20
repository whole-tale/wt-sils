(define (batch-set-alph inf outf)
	(let* 
		(
			(image (car (gimp-file-load 1 inf inf)))
			(drawable (car (gimp-image-get-active-layer image)))
		)
		(plug-in-colortoalpha  1 image drawable '(0 0 0))
		(gimp-file-save 1 image drawable outf outf)
		(gimp-image-delete image)
	)
)
