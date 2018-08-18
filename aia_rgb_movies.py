"""
After generating the RGB images with e.g. aia_rgb_images.py,
run any of the examples below to generate a movie with a given resolution and field of view.
"""
import visualization


# Directory of the rgb images
images_dir = '/Users/rattie/Data/SDO/AIA/event_2012_08_31/rgb'

if __name__ == '__main__':
    # Use uncropped images at 1/2 resolution (4096 x 4096 -> 2048 x 2048) - this only makes sense on 4k or 5k screens
    video_size = (2048, 2048)
    filename = 'rgb_movie_full_half'
    # Number of frames per second
    fps = 30
    # Encode movie
    visualization.encode_video(images_dir, filename, fps=fps, video_size=video_size)


    # Cropping over a 2048 x 2048 window in the bottom-left quadrant, at full resolution.
    crop = [2048, 2048, 0, 2048] # [width, height, x, y] where x, y are top-left corner of the cropping window
    video_size = (2048, 2048)
    filename = 'rgb_movie_crop_2048x2048_2048x2048'
    fps = 30
    # Encode movie
    visualization.encode_video(images_dir, filename, crop=crop, video_size=video_size)


    # Same as before at half resolution
    crop = [2048, 2048, 0, 2048] # [width, height, x, y] where x, y are top-left corner of the cropping window
    video_size = (1024, 1024)
    filename = 'rgb_movie_2048x2048_1024x1024'
    fps = 30
    # Encode movie
    visualization.encode_video(images_dir, filename, crop=crop, video_size=video_size)

    # With a 16:9 aspect ratio, crop over 3840 x 2160 around bottom half and output at full HD resolution (1920 x 1080)
    crop = [3840, 2160, 128, 1935]
    video_size = (1920, 720)
    filename = 'rgb_movie_3840x2160_1920x1080'
    fps = 30
    # Encode movie
    visualization.encode_video(images_dir, filename, crop=crop, video_size=video_size)

    # full sun rescaled to 1080x1080 and padded at 1920 x 1080 for optimized youtube videos
    frame_size = (1080, 1080)
    padding = (1920, 1080)
    filename = 'rgb_movie_full_padded_1920_1080'
    # Number of frames per second
    fps = 30
    # Encode movie
    visualization.encode_video(images_dir, filename, fps=fps, frame_size=frame_size, padding=padding)