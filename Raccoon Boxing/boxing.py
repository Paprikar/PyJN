def draw_boxes(img, boxes):
    import cv2
    
    for box in boxes:
        img = cv2.rectangle(img, (box['xmin'], box['ymin']), (box['xmax'], box['ymax']), (0, 255, 0), 2)
        cv2.putText(img, 
                    box['class']+' '+box['prob'] if 'brob' in box else box['class'], 
                    (box['xmin'], box['ymin'] - 6), 
                    cv2.FONT_ITALIC, 
                    1.2e-3 * img.shape[0], 
                    (0,255,0),
                    2,
                    cv2.LINE_AA)
    return img.astype('uint8')