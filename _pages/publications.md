---
layout: page
permalink: /publications/
title: publications
description: 
years: [2022, 2021, 2019, 2018, 2017, 2016, 2015]
nav: true
nav_order: 1
---

In reversed chronological order. You can also visit my [Google Scholar profile](https://scholar.google.com/citations?user=janVBUgAAAAJ) for more details.

<!-- _pages/publications.md -->
<div class="publications">

{%- for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @*[year={{y}}]* %}
{% endfor %}

</div>
