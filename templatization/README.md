This experiment looks at creating custom types (not that it has never been done before, but as they,
the case against reinventing the wheel is overrated). On more concrete grounds, the justification
lies in the unattractive realization that our simple C structures (aka, *x, malloc, cudaMalloc, etc.)
are becoming rather clunky in creating new experiments. The particular use case we have in mind is
in the examination of unsigned char for our histograms. Vectorizing uchar loads as uint (==4*uchar)
carries with it an obvious promise, asking for infrastructural developments we attempt to build in
the present iteration of our experiments. Our utilitarian psyche is perfectly in al(l)ignment with
those much frowned upon exploratory proclivities which I delude myself thinking might resonate
with the likes of Livingstone and Columbus. Into the the unexplored country from whose bourne,
every traveller must return, beckons. 