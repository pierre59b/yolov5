import argparse

import torch.backends.cudnn as cudnn

from utils import google_utils
from utils.datasets import *
from utils.utils import *

import pickle
import pandas as pd
import sys
import time
def detect(source = 'inference/images', folder=None, out = 'inference/output', save_img=False, weights= './models/yolov5s.pt', view_img=False, save_txt=False, imgsz= 640):
    # log_out = open("out.log", "a")
    # sys.stdout = log_out
    # log_err = open("err.log", "a")
    # sys.stderr = log_err

    # print("BEGIN")
    # sys.stderr.write( \
    #     "DETECT: "+folder)
    # print("FIRST DETECT"+folder)
    # if folder is not None:
    #     source += '/'+folder
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device('')
    if os.path.exists(out):
        pass
        # shutil.rmtree(out)  # delete output folder
    else:
        os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
    # model.fuse()
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # for i, n in enumerate(names):
        # print(str(i)+' '+n)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    df_results = pd.DataFrame(columns = ['youtube_id', 'class_id', 'object_id', 'image_nb', 'xmin', 'xmax', 'ymin', 'ymax', 'found_id', 'found_prec'])
    # print("DETECT: "+folder)
    # sys.stderr.write( \
        # "DETECT: "+folder)
    # print("before for")
    for path, img, im0s, vid_cap in dataset:
        # print("infor")
        
        img_name = path.split("/")[-1]
        # sys.stderr.write( \
            # "Detect: "+img_name)
        # print("Detect: "+img_name)
        if folder is not None:
            _, class_id, object_id = tuple(img_name.split("+"))
            object_id, image_nb = tuple(object_id.split("_"))
            image_nb = int(image_nb.split(".")[0])
            new_row = {'youtube_id': folder, 'class_id': class_id, 'object_id': object_id, 'image_nb': image_nb}
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        # print("before det")
        for i, det in enumerate(pred):  # detections per image
            # print("in det")
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # print("in det det")
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                if folder is not None:
                    for d in det:
                        # print("before row")
                        new_row.update({'xmin':d[0].item(), 'xmax':d[1].item(), 'ymin':d[2].item(), 'ymax':d[3].item(), 'found_id':int(d[-1].item()), 'found_prec':d[-2].item()})
                        # print("after row")
                        df_results = df_results.append(new_row, ignore_index = True)
                        # print("after append")
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
    if folder is not None:
        print('save in: ' + out+"/"+folder+".csv")
        df_results.to_csv(out+"/"+folder+".csv")
    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))
    time.sleep(2)


# if __name__ == '__main__':
#     source = 'yt_bb\\frames\\youtube_boundingboxes_detection_validation\\'
#     with torch.no_grad():
#         detect(source)

