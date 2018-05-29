% This is the MATLAB code for testing human performance on rapid visual
% object recognition under distorted conditions
% (To be compared against Deep Neural Network performance)

% Object task: choose the object that was shown

% This code is based on a related experiment of Tom Wallis (with his permission).
% It has been modified extensively by Robert Geirhos to fit the 
% experiment described above.


function object_recognition_experiment()
%% Parameters

disp(['Note: Please specify the experiment (colour, contrast, ...) IN the matlab script by changing the filepath to the corresponding .yaml file.']);

% get parameters from files:
filepathToHere=pwd;

yamlFileName =  'params_object_recognition_colour_experiment.yaml';


gen_file = fullfile(filepathToHere, yamlFileName);
params = ReadYaml(gen_file);

disp(['Note: This is the ', yamlFileName, ' file name for the parameters.']);

auditive_feedback = params.auditive_feedback;
visual_feedback = params.visual_feedback;


%% Set file paths and open file to write to; ask for subject code

data_path = fullfile(filepathToHere, sprintf('../../raw-data/%s/', num2str(params.experiment_name)));

if ~exist(data_path, 'dir')
    mkdir(data_path);
end

% use absolute paths to the stimuli images directory.
source_folder = 'object_recognition/stimuli/';  % path to /project/stimuli/ folder

response_im_path = fullfile(source_folder, 'response_screens/'); % the icons

expt_has_sessions = 0;
sess_num_entered = 1;

is_valid_answer1 = 0;
while ~is_valid_answer1
    subj = input('Enter subject code: ','s');
    
    is_valid_answer2 = 0;
    
    while ~is_valid_answer2
        session_expt_answer = input('Does this experiment have sessions (1 or more)? (y/n) ', 's');
        if strcmp(session_expt_answer, 'y')
            expt_has_sessions = 1;
            is_valid_answer2 = 1;
        elseif strcmp(session_expt_answer, 'n')
            expt_has_sessions = 0;
            is_valid_answer2 = 1;
        end
    end
    
    if expt_has_sessions
        sess_num_entered = str2double(input('Enter session number: ','s'));
        image_path = fullfile(source_folder, strcat(params.experiment_name, '/', subj, '/', 'session-', num2str(sess_num_entered)));
    else
        image_path = fullfile(source_folder, strcat(params.experiment_name, '/', subj, '/'));
    end
    
    if ~exist(image_path, 'dir')
        disp([image_path]);
        disp(['It seems there are no images for subject ', subj, ' --> Please try another subject / a different session!']);
    else
        is_valid_answer1 = 1;
    end
    
end

disp(['Check that viewing distance is ', num2str(params.dist_monitor), ' cm!!!'])


all_images = dir(image_path);
NUM_UNNECESSARY_DIRECTORIES = 2; % . and .. need to be removed
n_trials = length(all_images) - NUM_UNNECESSARY_DIRECTORIES;


%% Set practice params if subject == "practice"


if strcmp(subj, 'practice')

    n_trials_per_block = params.num_practice_trials_per_block;
    
    visual_feedback = 1;
    auditive_feedback = 1;
    
    practice = 1;
else
    practice = 0;
    n_trials_per_block = params.num_trials_per_block;
end


%% Start the hardware

% old version using lcd_gray -> we're not using this
old_version = 0;

if old_version
    % load gamma correction file:
    calib = load('your_calibration_file.mat');
end

if old_version
    clut = spline(calib.measurement, calib.input, linspace(0,1,(2^12))');
    clut(1:10) = 0;
    clut(clut<0)=0;
end

% LCD-initialization
if ~old_version
    win = window('lcd_color', 'bg_color', params.bg_color);
else
    win = window('lcd_gray', 'bg_color', params.bg_color);%, 'clut', clut);
end

aud_volume = 0.5;
aud = dpixx_audio_port('volume', aud_volume);
aud.create_beep('short_low', 'low', .15, 0.25);


%% Do session log (which session is this subject doing?)

datafilename = fullfile(data_path,...
    sprintf('%s_%s_session_%s.csv', num2str(params.experiment_name), subj, num2str(sess_num_entered)));


% Check for existing result file to prevent accidentally overwriting files
% from a previous session:
if fopen(datafilename, 'rt')~=-1
    fclose('all');
    
    if expt_has_sessions
        error('File already exists. Try different subject / session number.');
    else
        % no sessions - default case
        next_num_answer = input('File already exists. Do you want to use the next number? (y/n) ', 's');
        
        if strcmp(next_num_answer, 'y')
           while fopen(datafilename, 'rt') ~=-1
               sess_num_entered = sess_num_entered + 1;
               datafilename = fullfile(data_path,...
    sprintf('%s_%s_session_%s.csv', num2str(params.experiment_name), subj, num2str(sess_num_entered)));

           end
        else 
            error('File already exists. Try different subject.');
        end   
    end
end

% fill new variables in data structure:
for trial=1:n_trials
    dat_struct(trial).subj = subj;
    dat_struct(trial).session = sess_num_entered;
    dat_struct(trial).trial = nan;
    dat_struct(trial).rt = nan;
    dat_struct(trial).object_response = [];
    dat_struct(trial).category = [];
    dat_struct(trial).condition = [];
    dat_struct(trial).imagename = [];
end


%% Preload textures for object responses:

object_response_tex = zeros(length(params.final_cats_responses), 1);
for i = 1 : length(params.final_cats_responses)
    this_cat = params.final_cats_responses{i};
    [object_response_tex(i), ~, object_response_size] = make_image_texture_alpha(...
        sprintf('%s%s.png', response_im_path, this_cat));
end


%% Calculate spatial and temporal display workings.


n_frames_stim = ceil(params.ms_stim * win.framerate / 1000);
n_frames_fixate = ceil(params.ms_fixation * win.framerate / 1000);
n_frames_feedback = ceil(params.ms_feedback * win.framerate / 1000);
n_frames_mask = ceil(params.ms_mask * win.framerate / 1000);

% stimulus ramp:
stim_ramp_params.size = n_frames_stim;
stim_ramp = curve_tapered_cosine(stim_ramp_params);  % function from ishow.


% Position in the middle - no need to change if number of response categories
% changes.
fix_position = [(win.rect(3) / 2), win.rect(4) / 2];

% determine drawing rects:
fixation_rect_center = CenterRectOnPoint([0, 0, params.fixation_size, params.fixation_size], ...
    fix_position(1), fix_position(2));
surround_rect_center = CenterRectOnPoint([0, 0, params.surround_size, params.surround_size], ...
    fix_position(1), fix_position(2));

object_response_rect = [0, 0, object_response_size(2), object_response_size(1)];

idx = 1;
spacing = params.box_size + params.object_gap;  % distance between centres of boxes.
offset_row = [-1.5, -0.5, 0.5, 1.5];  % offset for each row
offset_col = [-1.5, -0.5, 0.5, 1.5];  % offset each col

NUM_COLS = 4;
NUM_ROWS = 4;

for col = 1 : NUM_COLS
    for row = 1 : NUM_ROWS
        this_cat = params.final_cats_responses{idx};
        x_pos = offset_col(col) * spacing + fix_position(1);
        y_pos = offset_row(row) * spacing + fix_position(2);
        object_rects.(this_cat) = CenterRectOnPoint(object_response_rect, ...
            x_pos, y_pos);
        idx = idx + 1;
    end
end


%% Trials
experiment_start_time = tic;


try
    ListenChar(2);
    
    for trial=1:n_trials
        
        fprintf('Trial %s of %s\n', num2str(trial), num2str(n_trials));
        
        % Save properties in data structure
        dat_struct(trial).trial = trial;
        
        if trial == 1
            reset_mouse_pos();
            win.draw_text('Click the mouse to start!');
            win.flip();
            wait_for_mouse_click();
            
            for itic = 1 : n_frames_fixate
                
                Screen('FrameRect',win.h,[255 255 255],fixation_rect_center);
                win.flip();
                
            end
        end
        
        HideCursor(win.h);
        
        %% Load parameters for this trial
        
        image_fname = all_images(trial+NUM_UNNECESSARY_DIRECTORIES).name;
        
        if iscell(image_fname)
            image_fname = image_fname{(1)};
        end
        
        
        % load image:
        fname = fullfile(image_path, image_fname);
        
        image_data = strsplit(fname, '_');
        condition_index = 5;
        category_index = 6;
                
        if (length(image_data) < 6) && (~ expt_has_sessions)
           error('It seems the experiment actually has a session number! Try restarting it again WITH entering a session number.'); 
        elseif length(image_data) < 6
           error('Image path cannot be split by _ in many parts as it should be the case.');
        end
        
        condition = image_data(condition_index);
        category = image_data(category_index);
        
        dat_struct(trial).condition = condition;
        dat_struct(trial).category = category;
        dat_struct(trial).imagename = image_fname;
             
        im = imread(fname);
        im = im2double(im);
        
        image_texture = win.make_texture(im);
        
        % generate mask:
        pink_mask = pinknoise_generate(200, 200);
        mask_texture = win.make_texture(pink_mask);
        
        %% wait ITI.
        
        for itic = 1 : n_frames_fixate
            %win.draw(asdf, stim_ramp(itic), surround_rect_center);
            Screen('FrameRect',win.h,[255 255 255],fixation_rect_center);
            win.flip();
        end
        
        
        %% stimulus presentation.
        
        for itic = 1 : n_frames_stim
            
            % centered image:
            win.draw(image_texture, stim_ramp(itic), ...
                surround_rect_center);

            Screen('DrawingFinished', win.h);
           
            win.flip();
        end
        
        % clear the screen:
        win.flip();
        
        
        %% Mask the image with pink noise
        
        %@R
        for itic = 1 : n_frames_mask
            
            win.draw(mask_texture, stim_ramp(itic), ...
                surround_rect_center);
            
            Screen('DrawingFinished', win.h);
            win.flip();
            
        end
        
        % close unused textures to save memory:
        Screen('Close', image_texture);
        
        % clear the screen:
        win.flip();
        
        
        %% Object identification response
        
        reset_mouse_pos();
        %%%%%% fixed timing response interval %%%%%%
        % Response interval
        response = [];
        rt = nan;
        
        % display response grid:
        win.draw(object_response_tex,...
            1, ...
            rect_struct_to_mat(object_rects));
        win.flip();
        start_resp = GetSecs;
        resp_time = GetSecs - start_resp;
        
        
        while resp_time < (params.ms_resp_object / 1000)
            
            % check mouse for clicks:
            [x, y, buttons] = get_mouse_pos();
            
            if any(buttons)
                possible_response = check_response_click(x, y, object_rects);
                if ~isempty(possible_response) % i.e., if response is valid.
                    % that way, the subject has the possibility to change its answer.
                    rt = GetSecs - start_resp;
                    response = possible_response;
                else
                    % not a valid click
                end
            end
            
            resp_time = GetSecs - start_resp;
        end
        
        if isempty(response)
            response = 'na';
        end
        
        disp(['Response: ', response, ' Correct: ', category]);
        
        dat_struct(trial).object_response = {response};
        dat_struct(trial).rt = rt;
        
        %% Display feedback if response != category (wrong response)
        
        if ~strcmp(category, response)
            
            if auditive_feedback
                aud.play('short_low');
            end
            
            if visual_feedback
                
                correct_rect = object_rects.(char(category));
                
                for itic = 1 : n_frames_feedback
                    
                    Screen('FillRect', win.h, 0.7, correct_rect);
                    win.draw(object_response_tex,...
                        1, ...
                        rect_struct_to_mat(object_rects));
                    win.flip();
                end
            end
            
        end
        
        HideCursor(win.h);
        
        %% inter-trial interval:
        for itic = 1 : n_frames_fixate
            
            %@R 3rd fixation presentation
            Screen('FrameRect',win.h,[255 255 255],fixation_rect_center);
            win.flip();
        end
        
        %% Breaks and summary feedback
        if mod(trial, n_trials_per_block)==0 && trial~=n_trials
            
            correct_sum = 0;
            for index = (trial - n_trials_per_block + 1):trial
                if ~strcmp(dat_struct(index).object_response, 'na')
                    if strcmp(dat_struct(index).object_response, dat_struct(index).category)
                        correct_sum = correct_sum + 1;
                    end
                end
            end
            
            obj_perc_corr = (correct_sum/n_trials_per_block)*100;
            
            msg = sprintf(['%d out of the last %d object identifications correct. '...
                'That corresponds to %.2f %% correct. \n'...
                'You have finished %d blocks out of %d. \n'...
                'PLEASE REMEMBER TO FIXATE! \n'...
                '\nClick a button to continue!'], ...
                correct_sum, n_trials_per_block, obj_perc_corr,...
                (trial/n_trials_per_block), round(n_trials/n_trials_per_block));
            
            
            win.draw_text(msg);
            win.flip();
            wait_for_mouse_click();
            
            
            for itic = 1 : n_frames_fixate
                
                %@R commented the following 2 lines:
                %win.draw_fixation(params.ppd, fix_position, fix_colour_oval, fix_colour_normal);
                %win.flip();
                
                Screen('FrameRect',win.h,[255 255 255],fixation_rect_center);
                win.flip();
                
            end
        end
        
        % Responded to last trial
        if trial == n_trials
            
            correct_sum = 0;
            
            if mod(trial, n_trials_per_block) == 0
                lower_index = (trial - n_trials_per_block + 1);
            else
                lower_index = (trial - (mod(trial, n_trials_per_block)) + 1);
            end
            
            for index = lower_index:trial
                if ~strcmp(dat_struct(index).object_response, 'na')
                    if strcmp(dat_struct(index).object_response, dat_struct(index).category)
                        correct_sum = correct_sum + 1;
                    end
                end
            end
            
            obj_perc_corr = (correct_sum/length(lower_index:trial))*100;
            
            msg = sprintf(['%d out of the last %d object identifications correct. '...
                'That corresponds to %.2f %% correct. \n'...
                'You have finished this experiment now. \n'...
                'Thanks a lot for your participation! \n'], ...
                correct_sum, length(lower_index:trial), obj_perc_corr);
            
            win.draw_text(msg);
            win.flip();
            WaitSecs(5);
        end
        
    end
    
    ListenChar(1);
    ShowCursor([], win.h);
    ShowCursor([], 0);
    
catch e
    ListenChar(1);
    ShowCursor([], win.h);
    ShowCursor([], 0);
    fclose('all');
    Screen('CloseAll');
    rethrow(e);
    
end

%% Write results to files:

if ~practice
    writetable(struct2table(dat_struct), datafilename);
end

fclose('all');

experiment_duration = toc(experiment_start_time) / 60;
fprintf('\nThe experiment took %2.1f minutes', experiment_duration);


%% Some subfunctions


    function [tex, im, sz] = make_image_texture(fname)
        im = imread(fname);
        im = im2double(im);
        sz = size(im);
        tex = win.make_texture(im);
    end

    function [tex, im, sz] = make_image_texture_alpha(fname)
        [im, ~, alpha] = imread(fname);  % image is an RGBA image.
        im(:, :, 4) = alpha;
        
        im = im2double(im);
        sz = size(im);
        tex = win.make_texture(im);
    end


% mouse response stuff:
    function [new_x, new_y, buttons] = get_mouse_pos()
        % return the current cursor position, checking to ensure
        % it's within the drawing window. Reset mouse position if
        % outside drawing window.
        [x, y, buttons] = GetMouse(win.h);
        new_x = max(min(x, win.rect(3)), win.rect(1));
        new_y = max(min(y, win.rect(4)), win.rect(2));
        if new_x ~= x || new_y ~= y
            SetMouse(new_x, new_y, win.h);
        end
    end

    function reset_mouse_pos()
        
        SetMouse(fix_position(1), fix_position(2), win.h);
        ShowCursor([], win.h);
        
    end

    function [x, y, buttons] = wait_for_mouse_click()
        [x,y,buttons] = GetMouse;
        while any(buttons) % if already down, wait for release
            [x,y,buttons] = GetMouse;
        end
        while ~any(buttons) % wait for press
            [x,y,buttons] = GetMouse;
        end
    end

    function response = check_response_click(x, y, rect_struct)
        entries = fields(rect_struct);
        response = [];
        for i = 1 : length(entries)
            if IsInRect(x, y, rect_struct.(entries{i}))
                response = entries{i};
            end
        end
    end

    function mat = rect_struct_to_mat(rect_struct)
        % returns a matrix of rects (4 rows by n cols)
        % from the structure of rects. Pass to win.draw.
        entries = fields(rect_struct);
        mat = zeros(4, length(entries));
        for i = 1 : length(entries)
            mat(:, i) = rect_struct.(entries{i})';
        end
    end


end
