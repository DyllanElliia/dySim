Gem::Specification.new do |spec|
  spec.name          = "DySim"
  spec.version       = "1.0"
  spec.authors       = ["DyllanElliia"]
  spec.email         = ["DyllanElliia_wzn@163.com"]

  spec.summary       = "Dyllan's Simulator"
  spec.license       = "GPL-3.0"
  spec.homepage      = "https://github.com/DyllanElliia/dySim"

  spec.files         = `git ls-files -z`.split("\x0").select { |f| f.match(%r!^(assets|_layouts|_includes|_sass|LICENSE|README)!i) }

  spec.add_runtime_dependency "github-pages", "~> 209"
end
