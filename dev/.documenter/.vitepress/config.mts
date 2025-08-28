import { defineConfig } from 'vitepress'
import { tabsMarkdownPlugin } from 'vitepress-plugin-tabs'
import mathjax3 from "markdown-it-mathjax3";
import footnote from "markdown-it-footnote";
import path from 'path'

function getBaseRepository(base: string): string {
  if (!base || base === '/') return '/';
  const parts = base.split('/').filter(Boolean);
  return parts.length > 0 ? `/${parts[0]}/` : '/';
}

const baseTemp = {
  base: '/GaussianMarkovRandomFields.jl/dev/',// TODO: replace this in makedocs!
}

const navTemp = {
  nav: [
{ text: 'Home', link: '/index' },
{ text: 'Tutorials', collapsed: false, items: [
{ text: 'Overview', link: '/tutorials/index' },
{ text: 'Building autoregressive models', link: '/tutorials/autoregressive_models' },
{ text: 'Spatial Modelling with SPDEs', link: '/tutorials/spatial_modelling_spdes' },
{ text: 'Spatiotemporal Modelling with SPDEs', link: '/tutorials/spatiotemporal_modelling' },
{ text: 'Boundary Conditions for SPDEs', link: '/tutorials/boundary_conditions' }]
 },
{ text: 'API Reference', collapsed: false, items: [
{ text: 'Overview', link: '/reference/index' },
{ text: 'GMRFs', link: '/reference/gmrfs' },
{ text: 'Latent Models', link: '/reference/latent_models' },
{ text: 'Observation Models', link: '/reference/observation_models' },
{ text: 'Gaussian Approximation', link: '/reference/gaussian_approximation' },
{ text: 'Hard Constraints', link: '/reference/hard_constraints' },
{ text: 'SPDEs', link: '/reference/spdes' },
{ text: 'Discretizations', link: '/reference/discretizations' },
{ text: 'Meshes', link: '/reference/meshes' },
{ text: 'Plotting', link: '/reference/plotting' },
{ text: 'Solvers', link: '/reference/solvers' },
{ text: 'Autoregressive Models', link: '/reference/autoregressive' },
{ text: 'Linear maps', link: '/reference/linear_maps' },
{ text: 'Preconditioners', link: '/reference/preconditioners' }]
 },
{ text: 'Bibliography', link: '/bibliography' },
{ text: 'Developer Documentation', collapsed: false, items: [
{ text: 'Overview', link: '/dev-docs/index' },
{ text: 'Solvers', link: '/dev-docs/solvers' },
{ text: 'SPDEs', link: '/dev-docs/spdes' },
{ text: 'Discretizations', link: '/dev-docs/discretizations' }]
 }
]
,
}

const nav = [
  ...navTemp.nav,
  {
    component: 'VersionPicker'
  }
]

// https://vitepress.dev/reference/site-config
export default defineConfig({
  base: '/GaussianMarkovRandomFields.jl/dev/',// TODO: replace this in makedocs!
  title: 'GMRFs.jl',
  description: 'Documentation for GaussianMarkovRandomFields.jl',
  lastUpdated: true,
  cleanUrls: true,
  outDir: '../1', // This is required for MarkdownVitepress to work correctly...
  head: [
    
    ['script', {src: `${getBaseRepository(baseTemp.base)}versions.js`}],
    // ['script', {src: '/versions.js'], for custom domains, I guess if deploy_url is available.
    ['script', {src: `${baseTemp.base}siteinfo.js`}]
  ],
  
  vite: {
    define: {
      __DEPLOY_ABSPATH__: JSON.stringify('/GaussianMarkovRandomFields.jl'),
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '../components')
      }
    },
    optimizeDeps: {
      exclude: [ 
        '@nolebase/vitepress-plugin-enhanced-readabilities/client',
        'vitepress',
        '@nolebase/ui',
      ], 
    }, 
    ssr: { 
      noExternal: [ 
        // If there are other packages that need to be processed by Vite, you can add them here.
        '@nolebase/vitepress-plugin-enhanced-readabilities',
        '@nolebase/ui',
      ], 
    },
  },
  markdown: {
    math: true,
    config(md) {
      md.use(tabsMarkdownPlugin),
      md.use(mathjax3),
      md.use(footnote)
    },
    theme: {
      light: "github-light",
      dark: "github-dark"}
  },
  themeConfig: {
    outline: 'deep',
    logo: { src: '/logo.svg', width: 24, height: 24},
    search: {
      provider: 'local',
      options: {
        detailedView: true
      }
    },
    nav,
    sidebar: [
{ text: 'Home', link: '/index' },
{ text: 'Tutorials', collapsed: false, items: [
{ text: 'Overview', link: '/tutorials/index' },
{ text: 'Building autoregressive models', link: '/tutorials/autoregressive_models' },
{ text: 'Spatial Modelling with SPDEs', link: '/tutorials/spatial_modelling_spdes' },
{ text: 'Spatiotemporal Modelling with SPDEs', link: '/tutorials/spatiotemporal_modelling' },
{ text: 'Boundary Conditions for SPDEs', link: '/tutorials/boundary_conditions' }]
 },
{ text: 'API Reference', collapsed: false, items: [
{ text: 'Overview', link: '/reference/index' },
{ text: 'GMRFs', link: '/reference/gmrfs' },
{ text: 'Latent Models', link: '/reference/latent_models' },
{ text: 'Observation Models', link: '/reference/observation_models' },
{ text: 'Gaussian Approximation', link: '/reference/gaussian_approximation' },
{ text: 'Hard Constraints', link: '/reference/hard_constraints' },
{ text: 'SPDEs', link: '/reference/spdes' },
{ text: 'Discretizations', link: '/reference/discretizations' },
{ text: 'Meshes', link: '/reference/meshes' },
{ text: 'Plotting', link: '/reference/plotting' },
{ text: 'Solvers', link: '/reference/solvers' },
{ text: 'Autoregressive Models', link: '/reference/autoregressive' },
{ text: 'Linear maps', link: '/reference/linear_maps' },
{ text: 'Preconditioners', link: '/reference/preconditioners' }]
 },
{ text: 'Bibliography', link: '/bibliography' },
{ text: 'Developer Documentation', collapsed: false, items: [
{ text: 'Overview', link: '/dev-docs/index' },
{ text: 'Solvers', link: '/dev-docs/solvers' },
{ text: 'SPDEs', link: '/dev-docs/spdes' },
{ text: 'Discretizations', link: '/dev-docs/discretizations' }]
 }
]
,
    editLink: { pattern: "https://github.com/timweiland/GaussianMarkovRandomFields.jl/edit/main/docs/src/:path" },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/timweiland/GaussianMarkovRandomFields.jl' }
    ],
    footer: {
      message: 'Made with <a href="https://luxdl.github.io/DocumenterVitepress.jl/dev/" target="_blank"><strong>DocumenterVitepress.jl</strong></a><br>',
      copyright: `© Copyright ${new Date().getUTCFullYear()}.`
    }
  }
})
