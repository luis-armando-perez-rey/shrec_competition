# Script to create a rendered images from the default camera position corresponding to a 3D scene in Sketchup. 
# To run the script open SketchUp command line and type 
# load /path_to_script/default_view_generation

# Rendering parameters
x_resolution = 256
y_resolution = 256

# Select directory
dir = UI.select_directory(title: "Select directory of .skp models")
files = Dir.entries(dir).select{|e| e =~ /[.]skp$/ }
puts(files)

# Saving directory 
save_dir = UI.select_directory(title: "Select directory for saving renders")

# View parameters
camera_view = Sketchup.active_model.active_view

for file in files
	skp_file_path = dir+"/"+file
	# Important constants
	puts("Loading "+skp_file_path)
	
	
	# Load sketchup model
	path = dir+"/"+file
	Sketchup.open_file(path)
	puts("Loaded "+skp_file_path)
	puts("Rendering view ")
	filename = file.split(".skp")[0]
	Sketchup.active_model.active_view.write_image({
	        :filename    => save_dir+"/"+filename+"_"+String(12)+"_.png",
        	:antialias   => true,
        	:transparent => true,
		:width => x_resolution,
		:height => y_resolution})
	Sketchup.active_model.close(ignore_changes = true)
end