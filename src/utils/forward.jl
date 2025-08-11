using MacroTools

"""
    @forward T.field f, g, h

Forward methods `f`, `g`, `h` to `T.field`. The macro creates method definitions that delegate
calls to the specified field's methods.

# Example
```julia
struct Wrapper{T}
    inner::T
end

@forward Wrapper.inner length, size, getindex

# Now these work automatically:
w = Wrapper([1, 2, 3])
length(w)  # equivalent to length(w.inner)
```
"""
macro forward(ex, fs)
  @capture(ex, T_.field_) || error("Syntax: @forward T.x f, g, h")
  T = esc(T)
  fs = isexpr(fs, :tuple) ? map(esc, fs.args) : [esc(fs)]
  :($([:($f(x::$T, args...; kwargs...) =
         (Base.@_inline_meta; $f(x.$field, args...; kwargs...)))
       for f in fs]...);
    nothing)
end