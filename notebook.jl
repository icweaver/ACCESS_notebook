### A Pluto.jl notebook ###
# v0.12.16

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
md"## $(@bind run_raw_stats CheckBox()) File stats"

# ╔═╡ 18536d9c-f922-11ea-02ed-5340b350c0c0
if run_raw_stats
	df_raw_data = CCDReduction.fitscollection(
		data_path,
		abspath=false,
		exclude=fn"ift*[!c1].fits",
		exclude_key=("COMMENT",)
	)
	
	# Save data
	header_keys = [
		"path",
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
	CSV.write(
		"$(project_path)/nightlog_$(basename(data_path)).csv",
		df_raw_data[header_keys]
	)
	
	# Plot a key
	key = "AIRMASS"
	plot(
		df_raw_data[key],
		label = "some data",
		linecolor = :cyan,
		#markershape = :square,
		markercolor = :cyan,
		legend = :none, #:bottomright,
		xlabel = "Index",
		ylabel = key,
		title = data_path,
	)

end

# ╔═╡ f802dde4-f92a-11ea-2bf6-bd8d80f69a3a
if run_raw_stats
	#by(df_raw_data, :OBJECT, nrow)
	HTMLTable(head(@where(df_raw_data, occursin.("science", :OBJECT))))
end

# ╔═╡ 0110f78a-f945-11ea-004c-291cbfd9365c
md"## $(@bind run_tepspec CheckBox()) `tepspec` exploration"

# ╔═╡ 362246e4-3a44-11eb-14b3-13a687b7f39b
const DATA_RED = "../Projects/HATP26b/data_reductions"

# ╔═╡ e7e183b0-fb4c-11ea-157d-c11e7dbb1a1c
md"#### $(@bind trace CheckBox()) Trace"

# ╔═╡ 93b51aba-fba4-11ea-2181-a9f87f9348e4
md"#### $(@bind extracted_spectra CheckBox()) Extracted spectra"

# ╔═╡ efde1340-3a57-11eb-10a8-63a5e2f99c30
md"""
plots items in `<object>_spec.fits` files \
has shape time x spec_item x ypix (|| to wav direction)
```julia
i =
0: Wavelength
1: Simple extracted object spectrum
2: Simple extracted flat spectrum
3: Pixel sensitivity (obtained by the flat)
4: Simple extracted object spectrum/pixel sensitivity
5: Sky flag (0 = note uneven sky, 1 = probably uneven profile,
   2 = Sky_Base failed)
6: Optimally extracted object spectrum
7: Optimally extracted object spectrum/pixel sensitivity
```
"""

# ╔═╡ 25af8e98-fba8-11ea-2089-db810bb8778a
function plot_spectra!(p, data; i=2, label=nothing)
	d = data[:, i+1, :]
	extracted_spectra_med = median(d, dims=2)
	σ = std(d, dims=2)
	plot!(
		p,
		extracted_spectra_med ./ maximum(extracted_spectra_med),
		ribbon=(
			σ / maximum(extracted_spectra_med),
			σ / maximum(extracted_spectra_med),
		),
		label = label,
		legend = :bottomleft,
	)
end

# ╔═╡ 9edfdb1e-fba4-11ea-2223-a5266f7ec2bb
if extracted_spectra
	p = plot()
	for fpath in glob("$DATA_RED/ut190313_a15_25_noflat_LBR/*spec.fits")
		FITS(fpath) do f
			data = read(f[1])
			plot_spectra!(p, data, label=basename(fpath), i=6)
		end
	end	
	p
end

# ╔═╡ 767b67be-3714-11eb-2a8b-e13213320300
md"### Detrending"

# ╔═╡ 586f547e-3714-11eb-0fa1-cf1a1ebd5f50
md"#### $(@bind run_GPT_WLC CheckBox()) GPT WLC"

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
	data = load_pickle(
		"$DATA_RED/ut190313_a15_25_noflat_LBR/LCs_hp26_bins.pkl"
	)
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
if trace
	XX = load_pickle("$DATA_RED/ut190313_a15_25_noflat_LBR/XX.pkl")
    YY = load_pickle("$DATA_RED/ut190313_a15_25_noflat_LBR/YY.pkl")
	
	md"""
	**Number of traces to show for:**
	$(@bind trace_key Select(collect([Pair(key,key) for key in keys(XX)])))
	$(@bind num_traces Slider(2:7, show_value=true))
	"""
end

# ╔═╡ 6d2b41fc-fb96-11ea-0ca4-1d62d94d25d7
if trace	
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

# ╔═╡ 6319a232-f8a3-11ea-0a58-7bf7af34ff4d
begin
	theme(:dark)
	default(
		titlefont = "Helvetica",
		guidefont = "Helvetica",
		markerstrokewidth = 0,
	)
end

# ╔═╡ 1497392c-f96d-11ea-3e29-15d8b60f4559
plotly()

# ╔═╡ Cell order:
# ╟─cea3f932-f935-11ea-0eb3-4f47e65da5fa
# ╟─18536d9c-f922-11ea-02ed-5340b350c0c0
# ╠═f802dde4-f92a-11ea-2bf6-bd8d80f69a3a
# ╟─0110f78a-f945-11ea-004c-291cbfd9365c
# ╠═362246e4-3a44-11eb-14b3-13a687b7f39b
# ╠═65530184-f968-11ea-3f55-7d8a470384ed
# ╟─6f8188e8-f98f-11ea-0bd4-bd30f24b976e
# ╟─51666c34-39bd-11eb-0e2a-2371e207dce8
# ╟─e7e183b0-fb4c-11ea-157d-c11e7dbb1a1c
# ╟─d6251ef2-fb4c-11ea-1ce3-97a485de4b95
# ╠═6d2b41fc-fb96-11ea-0ca4-1d62d94d25d7
# ╟─93b51aba-fba4-11ea-2181-a9f87f9348e4
# ╟─efde1340-3a57-11eb-10a8-63a5e2f99c30
# ╠═9edfdb1e-fba4-11ea-2223-a5266f7ec2bb
# ╠═25af8e98-fba8-11ea-2089-db810bb8778a
# ╟─767b67be-3714-11eb-2a8b-e13213320300
# ╟─586f547e-3714-11eb-0fa1-cf1a1ebd5f50
# ╟─8fbbec74-f970-11ea-34d8-174a293a4107
# ╟─4494f844-f98b-11ea-330d-e3c18965972d
# ╟─b38c37b6-f970-11ea-033e-6d215c8b87df
# ╟─8041e34c-f971-11ea-0a9e-dd6415b3f33c
# ╟─03cd4060-fab6-11ea-1298-ebbeccac9d09
# ╟─7b2ef6f4-f9e3-11ea-22ed-0367935634be
# ╠═6319a232-f8a3-11ea-0a58-7bf7af34ff4d
# ╟─1497392c-f96d-11ea-3e29-15d8b60f4559
# ╠═de8e285c-f93c-11ea-1571-a5a5dbc48e3c
