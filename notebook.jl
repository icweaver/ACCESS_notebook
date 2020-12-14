### A Pluto.jl notebook ###
# v0.12.17

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ de8e285c-f93c-11ea-1571-a5a5dbc48e3c
using PlutoUI, Plots, StatsPlots, CSV, CCDReduction, FITSIO, DataFrames, DataFramesMeta, Glob, AstroImages, HDF5, Dates, Printf, PyCall, RecursiveArrayTools, BenchmarkTools, Statistics, PaddedViews

# ╔═╡ cea3f932-f935-11ea-0eb3-4f47e65da5fa
md"""
## $(@bind run_file_stats CheckBox()) File stats

Processes raw data in `DATA_DIR` and saves summary of header info to file
"""

# ╔═╡ 30a59602-3e46-11eb-083b-01049a15e927
if run_file_stats
	const DATA_DIR = "./Projects/HATP26b/data/ut190313"
end

# ╔═╡ 18536d9c-f922-11ea-02ed-5340b350c0c0
if run_file_stats
	let
		# Load
		df = fitscollection(
			DATA_DIR,
			abspath=false,
			exclude=fn"ift*[!c1].fits",
			exclude_key=("COMMENT"),
		)

		# Save
		header_keys = [
			"FILENAME",
			"UT-DATE",
			"UT-TIME",
			"UT-END",
			"EXPTIME",
			"OBJECT",
			"EXPTYPE",
			"RA",
			"DEC",
			"SLITMASK",
			"FILTER",
			"DISPERSR",
			"AIRMASS",
			"G-SEEING",
			"BINNING",
			"SPEED",

		]
		csv_path = dirname(DATA_DIR)
		date = basename(DATA_DIR)
		CSV.write(
			"$csv_path/logs/nightlog_$date.csv",
			df[!, header_keys]
		)

		# Preview
		df
	end
end

# ╔═╡ 5b08e224-3e4c-11eb-358e-8532b35c3151
md"""
## $(@bind run_raw_data_plot CheckBox()) Raw data plot

View summary data stored in `LOG_PATH`
"""

# ╔═╡ 4bf3a536-3e4e-11eb-394e-6326142d4746
if run_raw_data_plot
	const LOG_PATH = "./Projects/HATP26b/data/logs/nightlog_ut190313.csv"
end

# ╔═╡ d9d441d8-3e48-11eb-3dda-f334426e5b80
if run_raw_data_plot
	let
		key = :AIRMASS
		df = CSV.read(LOG_PATH, DataFrame)
		#df = @where(df, occursin.("sci", :OBJECT))
		plot(
			df[!, key],
			label = "some data",
			linecolor = :cyan,
			#markershape = :square,
			markercolor = :cyan,
			legend = :none, #:bottomright,
			xlabel = "Index",
			ylabel = key,
			#title = data_path,
		)
	end
end

# ╔═╡ 0110f78a-f945-11ea-004c-291cbfd9365c
md"""
## $(@bind run_tepspec CheckBox()) `tepspec` exploration

Plot divided LCs and chip traces from `tepspec` pickle stored under `RED_DIR`
"""

# ╔═╡ ac13cca4-3e35-11eb-083b-cba8e8b46c07
begin
	const TEP_PKL = "./Projects/HATP26b/data_reductions/ut190313_a15_25_noflat_LBR/LCs_hp26_bins_r.pkl"
	const DATA_RED = dirname(TEP_PKL)
end

# ╔═╡ 93b51aba-fba4-11ea-2181-a9f87f9348e4
md"""
### $(@bind init_extracted_spectra CheckBox()) Intitial Extracted spectra

From spec.fits files
"""

# ╔═╡ 6b0c5d92-3bf7-11eb-03eb-871c9ffbd2e8
@bind extr_spec_key Select([
	"Wavelength",
	"Simple extracted object spectrum",
	"Simple extracted flat spectrum",
	"Pixel sensitivity (obtained by the flat)",
	"Simple extracted object spectrum/pixel sensitivity",
	"Sky flag (0 = note uneven sky, 1 = probably uneven profile,
		2 = Sky_Base failed)",
	"Optimally extracted object spectrum",
	"Optimally extracted object spectrum/pixel sensitivity",
		])

# ╔═╡ 25af8e98-fba8-11ea-2089-db810bb8778a
function plot_spectra!(p, data; key="Wavelength", label=nothing)
	i = Dict(
		"Wavelength" => 0,
		"Simple extracted object spectrum" => 1,
		"Simple extracted flat spectrum" => 2,
		"Pixel sensitivity (obtained by the flat)" => 3,
		"Simple extracted object spectrum/pixel sensitivity" => 4,
		"Sky flag (0 = note uneven sky, 1 = probably uneven profile,
		2 = Sky_Base failed)" => 5,
		"Optimally extracted object spectrum" => 6,
		"Optimally extracted object spectrum/pixel sensitivity" => 7,
	)[key]
	d = data[:, i+1, :]
	extracted_spectra_med = median(d, dims=2)
	σ = std(d, dims=2)
	plot!(
		p,
		extracted_spectra_med,
		ribbon = σ,
		label = label,
		legend = :topright,
	)
end

# ╔═╡ 9edfdb1e-fba4-11ea-2223-a5266f7ec2bb
if init_extracted_spectra
	let
		p = plot(xguide="Index", yguide="Value")
		for fpath in glob("$DATA_RED/*spec.fits")
			FITS(fpath) do f
				data = read(f[1])
				plot_spectra!(p, data, label=basename(fpath), key=extr_spec_key)
			end
		end	
		p
	end
end

# ╔═╡ a8e8d9e2-3c1a-11eb-2bc3-bd1f05686d01
md"""
### $(@bind final_extracted_spectra CheckBox()) Final Extracted spectra

Loaded from tepspec pickle
"""

# ╔═╡ 767b67be-3714-11eb-2a8b-e13213320300
md"## Detrending"

# ╔═╡ bcda6c82-3e53-11eb-34f6-81e71a34a293
const GPT_DET_WLC = "./Projects/HATP26b/data_detrending/out_r/HATP26/hp26_190313_r/white-light/PCA_2/detrended_lc.dat"

# ╔═╡ 586f547e-3714-11eb-0fa1-cf1a1ebd5f50
md"""### $(@bind run_GPT_WLC CheckBox()) GPT WLC

Visualize GPT data stored in `GPT_DET_WLC`
"""

# ╔═╡ 72c7e8f4-3cc3-11eb-2f6a-07aff33cd8aa
if run_GPT_WLC
	let 
		data = CSV.read(
			GPT_DET_WLC,
			DataFrame;
			comment = "#",
			header = ["Time", "DetFlux", "DetFluxErr", "Model"],
		)
		
		model = data.Model
		flux = data.DetFlux
		resid = (flux - model) * 1e6

		p_data = scatter(flux; label="data", yguide="Normalized flux")
		p_model_and_data = plot!(p_data, model; label="model", lw=4)
		p_resid = scatter(resid; label="residual", yguide="ppm")
		
		l = @layout [top; bottom{0.2h}]
		plot(p_model_and_data, p_resid; layout=l, link=:x, legend=nothing)
	end
end

# ╔═╡ c86f045e-3e56-11eb-191d-f76a1850b700
params = ["P", "aRs", "b"]

# ╔═╡ 968d4746-3e57-11eb-2811-7942a3f1bbb8
# @df df_GPT corrplot([:P :b :aRs :b])

# ╔═╡ 8fbbec74-f970-11ea-34d8-174a293a4107
md"### Photmetric monitoring"

# ╔═╡ 4494f844-f98b-11ea-330d-e3c18965972d
md"#### $(@bind run_phot_mon CheckBox()) ASAS-SN data"

# ╔═╡ b38c37b6-f970-11ea-033e-6d215c8b87df
if run_phot_mon
	df_phot = CSV.read(
		"projects/HATP23b/asas-sn_hatp23.csv",
	)
	HTMLTable(describe(df_phot))
end

# ╔═╡ 03cd4060-fab6-11ea-1298-ebbeccac9d09
function to_dt(s)
	# Converts y-m-d.x to y-m-dTh:m:s.x
	# Courtesy of @Yakir Luc Gagnon =]
	m = match(r"^(\d{4})-(\d\d)-(\d\d).(\d+)$", s)
	xs = parse.(Int, m.captures)
	x = pop!(xs)
	x /= exp10(ndigits(x))
	DateTime(xs...) + Millisecond(round(86_400_000*x))
end

# ╔═╡ 8041e34c-f971-11ea-0a9e-dd6415b3f33c
if run_phot_mon
	@df df_phot scatter(
		to_dt.(cols(Symbol("UT Date"))),
		cols(Symbol("flux(mJy)")),
		group = :Filter,
		legend = :bottomleft,
		xlabel = "UT Date",
		ylabel = "flux(mJy)",
	)
	
	transit_epochs = DateTime.(
		[
			"2016-06-22T08:18:00",
			"2017-06-10T07:05:00",
			"2018-06-04T07:24:00",
			"2018-06-21T06:56",
			"2018-08-22T03:30:00",
		]
	)
	vline!(transit_epochs, label="transits")
end

# ╔═╡ 7b2ef6f4-f9e3-11ea-22ed-0367935634be
# Call Julia's copy of Python to load in pickles
begin
	py"""
	import numpy as np
	import pickle
	
	def load_pickle(fpath):
		with open(fpath, "rb") as f:
			data = pickle.load(f)
		return data
	"""
	load_pickle = py"load_pickle"
end

# ╔═╡ 65530184-f968-11ea-3f55-7d8a470384ed
if run_tepspec
	data = load_pickle(TEP_PKL)
	flux_target = data["oLC"]
	flux_comps = data["cLC"]
	cNames = data["cNames"]
	with_terminal() do 
		println("Keys: ", keys(data))
		println("\ncomp stars: ", cNames)
	end
end

# ╔═╡ 6f8188e8-f98f-11ea-0bd4-bd30f24b976e
if run_tepspec
	md"""
	**comp star index:**
	$(@bind idx_comp Slider(1:length(cNames), show_value=true))
	"""
end

# ╔═╡ 51666c34-39bd-11eb-0e2a-2371e207dce8
if run_tepspec
	scatter(
			flux_target ./ flux_comps[:, idx_comp],
			label = "target / $(cNames[idx_comp])",
			linecolor = :cyan,
			markershape = :circle,
			markercolor = :cyan,
			legend = :topright,
			xlabel = "Index",
			ylabel = "Flux",
			title = "Divided WLC",
	)
end

# ╔═╡ d6251ef2-fb4c-11ea-1ce3-97a485de4b95
if run_tepspec
	XX = load_pickle("$DATA_RED/XX.pkl")
    YY = load_pickle("$DATA_RED/YY.pkl")
	
	md"""
	**Number of traces to show for:**
	$(@bind trace_key Select(collect([Pair(key,key) for key in keys(XX)])))
	$(@bind num_traces Slider(2:7, show_value=true))
	"""
end

# ╔═╡ 6d2b41fc-fb96-11ea-0ca4-1d62d94d25d7
if run_tepspec	
	trace_idxs = round.(Int, range(1, length(XX[trace_key]), length=num_traces))
	plot(
		XX[trace_key][trace_idxs],
		YY[trace_key][trace_idxs],
		label = reshape(trace_idxs, 1, :) .- 1,
		title = trace_key,
		legend = :left,
		legend_title = "trace 0-index",
		palette = palette(:diverging_isoluminant_cjo_70_c25_n256, num_traces),
	)
end

# ╔═╡ 07920dee-3c1e-11eb-2885-33fcf07a818f
if final_extracted_spectra
	let
		data = load_pickle(TEP_PKL)
		final_specs = data["optimal spectra"]
		wav_final_spec = final_specs["wavelengths"]
		p = plot()
		for (k, v) in final_specs
			if k != "wavelengths"
				flux_final_spec = vec(mean(v, dims=1))
				σ = vec(std(v, dims=1))
				if !any(isnan, flux_final_spec)
					plot!(
						p,
						wav_final_spec,
						flux_final_spec,
						ribbon = σ,
						label = k,
					)
				end
			end
		end
		p
	end
end

# ╔═╡ 03b4f076-3e56-11eb-133c-393802c43eb4
corner_pkl = load_pickle("./Projects/HATP26b/data_detrending/out_r/HATP26/hp26_190313_r/white-light/BMA_posteriors.pkl");

# ╔═╡ f3ad356e-3e56-11eb-0346-7f6399dc50bf
df_GPT = DataFrame(corner_pkl)[!, params]

# ╔═╡ 6319a232-f8a3-11ea-0a58-7bf7af34ff4d
begin
	theme(:dark)
	default(
		titlefont = "Lato",
		guidefont = "Lato",
		markerstrokewidth = 0,
	)
end

# ╔═╡ 1497392c-f96d-11ea-3e29-15d8b60f4559
plotly()

# ╔═╡ Cell order:
# ╟─cea3f932-f935-11ea-0eb3-4f47e65da5fa
# ╠═30a59602-3e46-11eb-083b-01049a15e927
# ╟─18536d9c-f922-11ea-02ed-5340b350c0c0
# ╟─5b08e224-3e4c-11eb-358e-8532b35c3151
# ╠═4bf3a536-3e4e-11eb-394e-6326142d4746
# ╟─d9d441d8-3e48-11eb-3dda-f334426e5b80
# ╟─0110f78a-f945-11ea-004c-291cbfd9365c
# ╠═ac13cca4-3e35-11eb-083b-cba8e8b46c07
# ╟─65530184-f968-11ea-3f55-7d8a470384ed
# ╟─6f8188e8-f98f-11ea-0bd4-bd30f24b976e
# ╟─51666c34-39bd-11eb-0e2a-2371e207dce8
# ╟─d6251ef2-fb4c-11ea-1ce3-97a485de4b95
# ╠═6d2b41fc-fb96-11ea-0ca4-1d62d94d25d7
# ╟─93b51aba-fba4-11ea-2181-a9f87f9348e4
# ╟─6b0c5d92-3bf7-11eb-03eb-871c9ffbd2e8
# ╟─9edfdb1e-fba4-11ea-2223-a5266f7ec2bb
# ╟─25af8e98-fba8-11ea-2089-db810bb8778a
# ╟─a8e8d9e2-3c1a-11eb-2bc3-bd1f05686d01
# ╟─07920dee-3c1e-11eb-2885-33fcf07a818f
# ╟─767b67be-3714-11eb-2a8b-e13213320300
# ╠═bcda6c82-3e53-11eb-34f6-81e71a34a293
# ╟─586f547e-3714-11eb-0fa1-cf1a1ebd5f50
# ╟─72c7e8f4-3cc3-11eb-2f6a-07aff33cd8aa
# ╠═03b4f076-3e56-11eb-133c-393802c43eb4
# ╠═c86f045e-3e56-11eb-191d-f76a1850b700
# ╠═f3ad356e-3e56-11eb-0346-7f6399dc50bf
# ╠═968d4746-3e57-11eb-2811-7942a3f1bbb8
# ╟─8fbbec74-f970-11ea-34d8-174a293a4107
# ╟─4494f844-f98b-11ea-330d-e3c18965972d
# ╠═b38c37b6-f970-11ea-033e-6d215c8b87df
# ╠═8041e34c-f971-11ea-0a9e-dd6415b3f33c
# ╟─03cd4060-fab6-11ea-1298-ebbeccac9d09
# ╟─7b2ef6f4-f9e3-11ea-22ed-0367935634be
# ╟─6319a232-f8a3-11ea-0a58-7bf7af34ff4d
# ╠═1497392c-f96d-11ea-3e29-15d8b60f4559
# ╠═de8e285c-f93c-11ea-1571-a5a5dbc48e3c
