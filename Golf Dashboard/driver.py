# Import convention
import streamlit as st
with st.spinner(text='Loading Dashboard...'):
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    import cv2
    import time
    from pose_utils import *



st.title('Dashboard')
df = pd.read_csv('../golf-db-original-videos/GolfDB.csv', index_col = 0)
df = df[(df.view == 'face-on') & (df.slow == 0)]
# st.write(df.columns)
st.table(df[['player','sex','club']].head(10))

col1, col2 = st.columns(2)
col1.title('Posture')
col2.title('Trajectory')


# Using 'with' notation:
with col1:
    # pass
    feed = st.selectbox('Feed',options=['Demo-Cam','Remote-Cam','Local', 'Upload', 'URL'], index=2)
    if feed == 'Demo-Cam':
        try:
            image = st.camera_input('Do a T-Pose!')
            if image:
                from pose_utils import *
                # Resize and pad the image to keep the aspect ratio and fit the expected size.
                input_image = tf.expand_dims(image, axis=0)
                input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

                # Run model inference.
                keypoints_with_scores = movenet(input_image)

                # Visualize the predictions with image.
                display_image = tf.expand_dims(image, axis=0)
                display_image = tf.cast(tf.image.resize_with_pad(
                    display_image, 1280, 1280), dtype=tf.int32)
                output_overlay = draw_prediction_on_image(
                    np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)

                plt.figure(figsize=(5, 5))
                plt.imshow(output_overlay)
                _ = plt.axis('off')
        except Exception as e:
            st.error(f'Feed not working :: {e}')
    elif feed == 'Remote-Cam':
        try: 
            ip = st.text_input('IP',value='192.168.0.101')
            port = st.number_input('Port',value=4747)
            video = st.video(f'http://{ip}:{port}/video')
        except Exception as e:
            st.error(f'Feed not working :: {e}')
    elif feed == 'Local':
        try: 
            # video = st.video(f'../golf-db-original-videos/videos_160/{id}.mp4')
            img = st.image('../Tiger_Woods.gif')
            # process = st.empty()
            # revert = st.empty()
            # process.button('Process')
            process = st.button('Process')
        except Exception as e:
            st.error(f'Instantiation not working :: {e}')
        if process:
            try:
                with st.spinner(text='Loading Data...'):
                    image_path = '../Tiger_Woods.gif'
                    image = tf.io.read_file(image_path)
                    image = tf.image.decode_gif(image)
                # Load the input image.
                    num_frames, image_height, image_width, _ = image.shape
                    crop_region = init_crop_region(image_height, image_width)
            except Exception as e:
                st.error(f'Data loader not working :: {e}')

            try:
                output_images = []
                bar = st.progress(0,'Processing...')
                for frame_idx in range(num_frames):
                    keypoints_with_scores = run_inference(movenet, image[frame_idx, :, :, :], crop_region,crop_size=[input_size, input_size])
                    output_images.append(draw_prediction_on_image(image[frame_idx, :, :, :].numpy().astype(np.int32),keypoints_with_scores, crop_region=None,close_figure=True, output_image_height=image_height))
                    crop_region = determine_crop_region(keypoints_with_scores, image_height, image_width)
                    bar.progress(np.round(frame_idx/(num_frames-1),2),'Processing...')
            except Exception as e:
                st.error(f'Inference not working :: {e}')
                
            try:
                # # Prepare gif visualization.
                with st.spinner():
                    output = np.stack(output_images, axis=0)
                    to_gif(output, duration=120)
                    bar.empty()
                    # process.empty()
                    # revert.button('Revert')
                    img.image('./animation.gif')
            except Exception as e:
                st.error(f'Visualtization not working :: {e}')
                
        # if revert:
        #     try:
        #         with st.spinner():
        #             img.image('../Tiger_Woods.gif')
        #             revert.empty()
        #             process.button('Process')
        #     except Exception as e:
        #         st.error(f'Revert not working :: {e}')
            
    elif feed == 'Upload':
        try:
            st.toast(f'PATH: ../golf-db-original-videos/videos_160/{id}.mp4')
            video_file = st.file_uploader('Upload Video',type=['mp4'])
            if video_file is not None:
                st.video(video_file)
            else:
                st.warning('Please upload a video file.')
        except Exception as e:
            st.error(f'Feed not working :: {e}')
    else:
        try: 
            # url = https://www.youtube.com/shorts/PSQjAKSeXXc?feature=share
            # downloader = 'https://en.savefrom.net/391GA/'
            url = st.text_input('URL', value='https://rr3---sn-cvh7kney.googlevideo.com/videoplayback?expire=1694101151&ei=P5r5ZOPiI-zZx_AP55uy0Aw&ip=154.6.130.142&id=o-AMmioTK5yE9IT-qtyEe3hruihFSBjQVLmr3rbcv4BpTw&itag=136&source=youtube&requiressl=yes&spc=UWF9f-hOiR2NmObSq9myvSiZdvJrYek&vprv=1&svpuc=1&mime=video%2Fmp4&gir=yes&clen=514160&dur=12.199&lmt=1655143305252905&keepalive=yes&fexp=24007246,24350017,24362685,51000022&beids=24350017&c=ANDROID&txp=6319224&sparams=expire%2Cei%2Cip%2Cid%2Citag%2Csource%2Crequiressl%2Cspc%2Cvprv%2Csvpuc%2Cmime%2Cgir%2Cclen%2Cdur%2Clmt&sig=AOq0QJ8wRQIhAPD7OOMmge5qcpljp7RNgsrC9icsckxvNzVuTZDU03NUAiAdur64P_t9D_JOPiwT4bKZTb7BhTMkCWnl_7CV7ZsLdQ%3D%3D&rm=sn-4g5ere76&req_id=cd8e1a024b8ca3ee&redirect_counter=2&cm2rm=sn-5aap5ojx-jj0z7z&cms_redirect=yes&cmsv=e&ipbypass=yes&mh=Zr&mip=116.206.202.141&mm=29&mn=sn-cvh7kney&ms=rdu&mt=1694078985&mv=m&mvi=3&pl=24&lsparams=ipbypass,mh,mip,mm,mn,ms,mv,mvi,pl&lsig=AG3C_xAwRQIhANgjz1b8kPIo9cOHmzl5590Kc8tLgTQ_wfYzaFPbqXRZAiB3OXxkPvEkUBAPbk6X_F2aM241muvwYM869U1EhRAGRg%3D%3D')
            video = st.video(url)
        except Exception as e:
            st.error(f'Feed not working :: {e}')
            
            
with col2:
    from trajectory_utils import golf_ball_trajectory
    velocity, spin = st.columns(2)
    with velocity:
        st.info('Velocity')
        Vx = st.number_input('Vx',value=25)
        Vy = st.number_input('Vy',value=25)
        Vz = st.number_input('Vz',value=25)
    with spin:
        st.info('Spin')
        Ax = st.number_input('Ax',value=0)
        Ay = st.number_input('Ay',value=200)
        Az = st.number_input('Az',value=0)

    # trajectory,duration,fig = golf_ball_trajectory([25,25,25],[0,200,0])
    trajectory,duration,apex,carry,fig = golf_ball_trajectory([Vx,Vy,Vz],[Ax,Ay,Az])
    st.pyplot(fig)
    st.success(f'Duration: {duration:.1f}s, Apex: {apex:.1f}m, Carry: {carry:.1f}m')
    
st.title('Angles')
option = st.selectbox('Select Player:', options=[f'{i} ({j})' for i,j in zip(df.player, df.id)], index=0)
id = int(option.split(' ')[-1].lstrip('(').rstrip(')'))
st.table(df[df.id == id][['player','sex','club']])
st.write('KEYPOINT_DICT')
st.write(KEYPOINT_DICT)
# address, toe_up, mid_backswing, top, mid_downswing, impact, mid_follow_through, finish = 
# events = [address, toe_up, mid_backswing, top, mid_downswing, impact, mid_follow_through, finish]

cols = list(st.columns(2))
angle_points=['5-7-9','7-9-11','9-11-13','11-13-15','6-8-10','8-10-12','10-12-14','12-14-16']
event_counter = 0
for event in SWING_SEQUENCE:
    image_path = f'../golf-db-entire-image/Swing_events/{event}/{id}.jpg'

    try:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image)

        input_image = tf.expand_dims(image, axis=0)
        input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
    except Exception as e:
        st.error(f'Loading Error: {e}')
    
    # Run model inference.
    try:
        keypoints_with_scores = movenet(input_image)
    except Exception as e:
        st.error(f'Inference Error: {e}')
        
    # Visualize the predictions with image.
    try:
        display_image = tf.expand_dims(image, axis=0)
        display_image = tf.cast(tf.image.resize_with_pad(display_image, 1280, 1280), dtype=tf.int32)
        output_overlay = draw_prediction_on_image(np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)
    except Exception as e:
        st.error(f'Overlay Error: {e}')
        
    with cols[(event_counter % 2)]:
        try:
            st.image(output_overlay,caption=event)
        except Exception as e:
            st.error(f'Display Error: {e}')
        # plt.figure(figsize=(5, 5))
        # plt.imshow(output_overlay)
        # _ = plt.axis('off')

        try:
            angles[event] = [angle_caculator(keypoints_with_scores[0][0][5][:-1],keypoints_with_scores[0][0][7][:-1],keypoints_with_scores[0][0][9][:-1]),
                            angle_caculator(keypoints_with_scores[0][0][7][:-1],keypoints_with_scores[0][0][9][:-1],keypoints_with_scores[0][0][11][:-1]),
                            angle_caculator(keypoints_with_scores[0][0][9][:-1],keypoints_with_scores[0][0][11][:-1],keypoints_with_scores[0][0][13][:-1]),
                            angle_caculator(keypoints_with_scores[0][0][11][:-1],keypoints_with_scores[0][0][13][:-1],keypoints_with_scores[0][0][15][:-1]),
                            angle_caculator(keypoints_with_scores[0][0][6][:-1],keypoints_with_scores[0][0][8][:-1],keypoints_with_scores[0][0][10][:-1]),
                            angle_caculator(keypoints_with_scores[0][0][8][:-1],keypoints_with_scores[0][0][10][:-1],keypoints_with_scores[0][0][12][:-1]),
                            angle_caculator(keypoints_with_scores[0][0][10][:-1],keypoints_with_scores[0][0][12][:-1],keypoints_with_scores[0][0][14][:-1]),
                            angle_caculator(keypoints_with_scores[0][0][12][:-1],keypoints_with_scores[0][0][14][:-1],keypoints_with_scores[0][0][16][:-1])]
            
            st.write({points:f'{values:.2f} Rad' for points,values in zip(angle_points,angles[event])})
        except Exception as e:
            st.error(f'Writing Error: {e}')
    
    event_counter += 1