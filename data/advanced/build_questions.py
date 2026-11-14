"""Source of the advanced benchmark (data/advanced/questions.json). Run: python data/advanced/build_questions.py

Document: World Bank (2022), "Vietnam Country Climate and Development Report", 100 physical pages.
`page` is the physical PDF page (1-based); `printed_page` is the number printed on the page
(physical = printed + 10 for the body). Evidence was read from page images rendered from the
original PDF, not from model-extracted text.

Each answerable question has one main `group` and `evidence_groups`: a list of groups, each a
list of alternative evidence items. The context is sufficient when every group has at least one
item retrieved (same page; for a table/figure, a chunk of that element or a text chunk of that page).
The split into dev and test was fixed before any retrieval run.
"""
import json
from pathlib import Path


def ev(page, element, evidence):
    return {'page': page, 'printed_page': page - 10, 'element': element, 'evidence': evidence}


def q(id, split, group, question, answer, groups, calc=None):
    item = {'id': id, 'split': split, 'group': group, 'answerable': True, 'question': question,
            'expected_answer': answer, 'evidence_groups': groups}
    if calc:
        item['calculation'] = calc
    return item


def u(id, split, question, check):
    return {'id': id, 'split': split, 'group': 'unanswerable', 'answerable': False,
            'question': question, 'expected_behavior': 'search_and_abstain', 'absence_reason': check}


GREP = ('Checked on 2026-11-14: regex over the pypdf text of all 100 pages, plus reading the pages '
        'that mention the topic. ')

QUESTIONS = [
    # ------------------------------------------------------------------ dev
    q('ad01', 'dev', 'table', 'In the cross-country emission overview, what is Thailand\'s carbon intensity?',
      '851.4 tCO2e per million $ of GDP.', [[ev(18, 'Table 1', 'Thailand row, column "Carbon intensity (tCO2e/million $GDP)" = 851.4')]]),
    q('ad02', 'dev', 'table', 'What is the existing financing for adaptation, as a share of GDP per year and in accumulated NPV terms for 2022-2050?',
      '1.8 percent of GDP per year; $136.9 billion in NPV terms.', [[ev(29, 'Table 2', 'row "Existing financing": 1.8 (% of GDP per year), 136.9 ($ billion NPV)')]]),
    q('ad03', 'dev', 'table', 'Under the net-zero pathway with reduced air pollution, how would GDP change in 2030 relative to the baseline?',
      'GDP would be 0.8 percent higher in 2030.', [[ev(48, 'Table 4', 'row "NZP with reduced air pollution", % change in GDP 2030 = 0.8'),
                                                 ev(46, 'text', 'that would result in GDP being 0.8 percent higher in the NZP in 2030 than in the BAU scenario')]]),
    q('ad04', 'dev', 'table', 'In the Accelerated Decarbonization Scenario, how much more would be spent on solar capital costs than in the baseline (CPS), in dollars and in percent?',
      '+$21 billion, a 429 percent increase.', [[ev(52, 'Table 5', 'row "Solar": +21 ($ billion), +429% change')]]),
    q('ad05', 'dev', 'table', 'In the prioritization criteria, which combination of urgency and impact describes "quick wins"?',
      'High urgency and high impact: actions that cannot be postponed and will produce a high and immediate impact.', [[ev(78, 'Table 6', 'cell High urgency x High impact: "Actions that cannot be postponed and will produce a high and immediate impact (quick wins)"')]]),
    q('ad06', 'dev', 'table', 'What was Vietnam\'s government effectiveness percentile rank in 2020?',
      '61.54 (percentile rank, 2020, WGI).', [[ev(86, 'Annex 1 table', 'row "Government effectiveness", 2020 = 61.54, source WGI')]]),
    q('ad07', 'dev', 'image', 'On the map of extreme hazard exposure of national roads, in which parts of the country is landslide exposure concentrated?',
      'In the northern mountainous region and along the central part of the country (central highlands/central provinces); the south shows little landslide exposure.',
      [[ev(34, 'Figure 9', 'left map "Landslide": orange shading in the north and a dark band along central Vietnam; south grey')]]),
    q('ad08', 'dev', 'image', 'According to the financing needs diagram, how much would come from public sources and what share of GDP per year is that?',
      'Public: $130 billion, about 2.4 percent of GDP per year.', [[ev(71, 'Figure 21', 'middle box "PUBLIC $130 billion ~2.4% of GDP per year"')]]),
    q('ad09', 'dev', 'chart', 'What share of businesses reported that climate change caused "a lot" of business or production interruption in 2019?',
      '16 percent.', [[ev(23, 'Figure 6', 'bar "Business/production interruption": Not at all 11, Just a little 35, Quite a lot 38, A lot 16')]]),
    q('ad10', 'dev', 'chart', 'Which measure to reduce GHG emissions from rice production is the most expensive per tonne of CO2e, and roughly what does it cost?',
      'Moving from rice to maize, at roughly $390 per tCO2e (about 400 on the chart).', [[ev(59, 'Figure 16', 'tallest bar "Moving from rice to maize", just under 400 $/tCO2e')]]),
    q('ad11', 'dev', 'chart', 'Which livestock measure has the highest cost per tonne of CO2e reduced?',
      'Improving quality of diet for buffaloes (about $700 per tCO2e).', [[ev(60, 'Figure 17', 'tallest bar "Improving quality of diet for buffaloes", about 700 $/tCO2e')]]),
    q('ad12', 'dev', 'chart', 'By how many points would the NZP change inequality (Gini index) by 2040 relative to BAU?',
      'It would reduce the Gini index by 1.1 points.', [[ev(64, 'Figure 18', 'bar "Inequality (Gini Index)" = -1.1 points deviation from BAU')]]),
    q('ad13', 'dev', 'chart', 'What are power sector GHG emissions in 2040 under CPS and under ADS?',
      'About 246 MtCO2e under CPS and 63 MtCO2e under ADS.', [[ev(51, 'Figure 13', 'right panel end labels: CPS 246, ADS 63 (MtCO2e, 2040)')]]),
    q('ad14', 'dev', 'multi-hop', 'How do electricity/heat emissions in 2018 compare with power sector emissions projected for 2040 under the current policy scenario (CPS)?',
      'Electricity/heat emitted 109.13 MtCO2e in 2018; CPS power emissions reach about 246 MtCO2e in 2040, roughly 2.3 times higher.',
      [[ev(85, 'Annex 1 table', 'row "Electricity/heat" 2018 = 109.13 MtCO2e')], [ev(51, 'Figure 13', 'CPS 2040 = 246 MtCO2e')]], '246 / 109.13 = 2.25'),
    q('ad15', 'dev', 'multi-hop', 'What share of GDP per year is needed for the decarbonizing pathway, and how much do the NZP investment needs add up to for 2022-2040?',
      'Decarbonizing: 2.1 percent of GDP per year; NZP investment needs total $81.3 billion (NPV) for 2022-2040.',
      [[ev(71, 'Figure 21', '"Decarbonizing: 2.1% of GDP per year"')], [ev(46, 'Table 3', 'NZP investment needs 2022-2040 = 81.3'), ev(45, 'text', 'estimated at about $81.3 billion in net present value terms between 2022 and 2040')]]),
    q('ad16', 'dev', 'text', 'What share of motorized vehicles in Vietnam are motorbikes?', '93.3 percent.',
      [[ev(57, 'text', 'Motorbikes make up 93.3 percent of motorized vehicles in Vietnam')]]),
    q('ad17', 'dev', 'text', 'How large are the daily economic losses if a single link on National Highway 1A becomes unavailable?',
      'About $1.9 million per day, only from rerouting freight (excluding damage to the infrastructure).', [[ev(34, 'text', 'could cause $1.9 million per day in economic losses just from the need to reroute freight flows')]]),
    q('ad18', 'dev', 'text', 'How many people in Vietnam were estimated in 2020 to be directly exposed to intense flood risks?',
      '11.8 million people.', [[ev(38, 'text', 'that 11.8 million people in Vietnam are directly exposed to intense flood risks')]]),
    q('ad19', 'dev', 'text', 'How did electric vehicle sales in Vietnam develop between 2010 and 2020?',
      'From 1,579 in 2010 to about 1.9 million in 2020 (seven percent of private vehicle sales), mainly e-motorbikes and e-bicycles.',
      [[ev(58, 'text', 'sales in Vietnam grew from just 1,579 in 2010 to about 1.9 million in 2020, or seven percent of private vehicle sales')]]),
    u('ud01', 'dev', 'How many electric cars were sold in Vietnam in 2021?',
      GREP + "EV sales are given only for 2010 and 2020 (p58: 1,579 and 1.9 million, mostly e-motorbikes and e-bicycles); 'electric car' appears only as a measure label in Figure 15 (p57). No 2021 figure and no car-only figure."),
    u('ud02', 'dev', 'How many million tonnes of LNG per year does Vietnam plan to import by 2030?',
      GREP + "'LNG' -> 1 page (p56), which discusses policy reforms and selectivity for LNG facilities with no import volume or year target."),
    u('ud03', 'dev', 'How many hectares of mangroves did Vietnam replant between 2015 and 2020?',
      GREP + "'hectare|ha' -> pages 8 (a name), 60 (AWD rice, VND per ha), 61 (forest value per hectare); 'mangrove' -> 10 pages, none with an area or replanting figure."),

    # ------------------------------------------------------------------ test
    q('at01', 'test', 'table', 'In the cross-country emission overview, what are Indonesia\'s total emissions and its share of global emissions?',
      '1,703.9 MtCO2e, or 3.6 percent of global emissions.', [[ev(18, 'Table 1', 'Indonesia row: total emissions 1703.9 MtCO2e, % of global emissions 3.6%')]]),
    q('at02', 'test', 'table', 'What are the estimated financing needs for financial support to affected people and businesses for 2022-2050?',
      '0.3-0.7 percent of GDP per year, or $22.8-53.2 billion accumulated in NPV terms.', [[ev(29, 'Table 2', 'row "Financial support to affected people and businesses": 0.3-0.7, 22.8-53.2')]]),
    q('at03', 'test', 'table', 'Under the NZP with supporting reforms, what are the output gains and the net economic impact for 2031-2040?',
      'Output gains of $68.7 billion and a net economic impact of +$19.3 billion (NPV).', [[ev(46, 'Table 3', 'NZP with supporting reforms, 2031-2040: Output gains 68.7, Net economic impact 19.3')]]),
    q('at04', 'test', 'table', 'Apart from combining all supporting measures, which single supporting policy would leave GDP higher than the baseline in 2040 under the NZP, and by how much?',
      'Energy efficiency gains: GDP +0.2 percent in 2040 (all other single policies are negative in 2040).',
      [[ev(48, 'Table 4', '% change in GDP 2040: reduced air pollution -1.0, competitiveness -0.6, energy efficiency 0.2, labor mobility -0.2, low crowding out -0.9; all measures 3.3')]]),
    q('at05', 'test', 'table', 'How much would capital costs for offshore wind change in the Accelerated Decarbonization Scenario compared with the baseline?',
      'Up by $39 billion, a 402 percent increase.', [[ev(52, 'Table 5', 'row "Wind (offshore)": +39 ($ billion), +402%')]]),
    q('at06', 'test', 'table', 'What were Vietnam\'s per capita and total electricity consumption in 2020?',
      '2,745 kWh per capita and 267.18 TWh in total.', [[ev(85, 'Annex 1 table', 'Per capita electricity consumption 2020 = 2745 kWh; Total electricity consumption 2020 = 267.18 TWh')]]),
    q('at07', 'test', 'image', 'In the diagram of the new development paradigm, what actions are listed at the centre, between "Vietnam Today" and "Vietnam in 2045", and which two pathways connect them?',
      'Building resilient infrastructure; protecting the most vulnerable; implementing structural reforms to incentivize farmers and firms; pricing carbon; mobilizing financing. The Resilient Pathway and the Decarbonizing Pathway.',
      [[ev(12, 'Figure 1', 'centre list of five bullets; left arrow "Resilient Pathway", right arrow "Decarbonizing Pathway"')]]),
    q('at08', 'test', 'image', 'On the map of extreme hazard exposure of national roads, which hazard affects roads mainly along the coast?',
      'Typhoon flooding (right-hand map), with exposure along the northern and central coastline.', [[ev(34, 'Figure 9', 'right map "Typhoon flooding": green shading along the coast only')]]),
    q('at09', 'test', 'image', 'On the map of flood risks in coastal regions, which of the labeled cities lies in the zone with the highest relative flood risk?',
      'Da Nang, on the central coast (darkest shading).', [[ev(38, 'Figure 11', 'labeled cities Hanoi, Haiphong, Da Nang, Ho Chi Minh City, Can Tho; darkest green band on the central coast around Da Nang')]]),
    q('at10', 'test', 'image', 'According to the map of domestic coal production, which province is the major coal producing province and which two places in it are marked?',
      'Quang Ninh, with Hongai and Campha marked.', [[ev(54, 'Figure 14', 'box "Quang Ninh - Major Coal Producing Province", red circles Hongai and Campha')]]),
    q('at11', 'test', 'image', 'In the diagram of financing needs and sources for 2022-2040, how much is expected from private sources and what share of GDP per year is that?',
      'Private: $184 billion, about 3.4 percent of GDP per year.', [[ev(71, 'Figure 21', 'left box "PRIVATE $184 billion ~3.4% of GDP per year"')]]),
    q('at12', 'test', 'chart', 'Which category had the second-largest estimated cost of climate change in 2020, and how large was it?',
      'Labor productivity, $2,807 million (after fishery, $4,219 million).', [[ev(17, 'Figure 3', 'treemap: Fishery 4219, Labor Productivity 2807, Sea Level Rise 1474, Agriculture 837, ...')]]),
    q('at13', 'test', 'chart', 'What was the SOEs\' share of output in the steel sector in 2018?',
      '14 percent.', [[ev(24, 'Figure 7', 'line end label "Steel, 14%" at 2018 (prel.)')]]),
    q('at14', 'test', 'chart', 'How much electricity would coal generate in 2040 under CPS and under ADS?',
      'About 276 TWh under CPS and 26 TWh under ADS.', [[ev(51, 'Figure 13', 'left panel coal segments: BAU 550, CPS 276, ADS 26 TWh')]]),
    q('at15', 'test', 'chart', 'Under the NZP with reforms, which sector would see the largest employment increase in 2040, and by how much?',
      'Electricity, +23.6 percent relative to BAU.', [[ev(66, 'Figure 20', 'NZP with reforms bars: Electricity 23.6, Mining 18.4, Services 12.4, ...')]]),
    q('at16', 'test', 'chart', 'How did outstanding green credit change from 2016 to 2020?',
      'It rose from $3.7 billion in 2016 to $14.5 billion in 2020.', [[ev(72, 'Figure 22', 'Green credit outstanding bars: 2016 3.7, 2017 7.8, 2018 10.3, 2019 13.9, 2020 14.5, 2021:Q1 14.5 ($ billion)')]]),
    q('at17', 'test', 'multi-hop', 'Roughly how much electricity did coal generate in 2020, and how does that compare with coal generation in 2040 under the current policy scenario (CPS)?',
      'About 118 TWh in 2020 (59 percent of 200 TWh); under CPS coal would generate about 276 TWh in 2040, more than twice as much.',
      [[ev(49, 'Figure 12', 'power produced 2020 = 200 TWh, coal share 59%'), ev(49, 'text', 'coal power accounted for ... 59 percent of electricity generation')],
       [ev(51, 'Figure 13', 'CPS coal 2040 = 276 TWh')]], '0.59 x 200 = 118 TWh; 276 / 118 = 2.3'),
    q('at18', 'test', 'multi-hop', 'How much would the $90 carbon tax option (with sectoral investments) cut emissions by 2030, and how would GDP change in 2030 if the NZP were combined with all supporting measures?',
      'Emissions 42.8 percent below BAU by 2030; GDP 1.7 percent higher than BAU in 2030.',
      [[ev(45, 'text', 'the $90 tax would reduce emissions by 42.8 percent relative to the BAU scenario by 2030')],
       [ev(48, 'Table 4', 'NZP with all supporting measures, GDP 2030 = 1.7'), ev(48, 'text', 'while increasing GDP by 1.7 percent in 2030')]]),
    q('at19', 'test', 'multi-hop', 'How do transport GHG emissions in 2018 compare with what they are projected to reach by 2030 under BAU?',
      '36.32 MtCO2e in 2018; nearly 90 MtCO2e by 2030, about 2.5 times higher.',
      [[ev(85, 'Annex 1 table', 'row "Transportation" 2018 = 36.32 MtCO2e')], [ev(56, 'text', 'GHG emissions from transport will grow by 6-7 percent per year to 2030, reaching nearly 90 MtCO2e')]], '90 / 36.32 = 2.5'),
    q('at20', 'test', 'multi-hop', 'How does the current Environmental Protection Tax on gasoline compare with the starting level of the most modest economy-wide carbon tax option analyzed?',
      'EPT on gasoline is about $77.60 per tCO2e, while the most modest option starts at about $12 per tCO2e in 2022 (rising to $40 by 2040), so gasoline is taxed about 6.5 times the starting level.',
      [[ev(22, 'text', 'is around $0.50 per tCO2e on coal, $77.60 per tCO2e on gasoline, and $32.90 per tCO2e on diesel')],
       [ev(45, 'text', 'The most modest option would start at about $12 per tCO 2e in 2022 ... ramp up to $40 per tCO2e by 2040')]], '77.60 / 12 = 6.5'),
    q('at21', 'test', 'text', 'How does Vietnam rank on the ND-GAIN index and on the Germanwatch Global Climate Risk Index?',
      '127th out of 182 on ND-GAIN, and 13th highest among 180 countries on the Germanwatch index for 2000-19.', [[ev(16, 'text', 'ranked 127 out of 182 by the Notre Dame Global Adaptation Initiative (ND-GAIN), and 13th highest among 180 countries by the Germanwatch Global Climate Risk Index for 2000-19')]]),
    q('at22', 'test', 'text', 'What share of Ca Mau and Kien Giang provinces would a one-meter sea-level rise inundate?',
      '79 percent of Ca Mau and 76 percent of Kien Giang.', [[ev(37, 'text', 'A meter of sea-level rise would inundate 79 percent of the total area of Ca Mau province and 76 percent of Kien Giang province')]]),
    q('at23', 'test', 'text', 'According to the International Rice Research Institute model, how much would investing in AWD and related techniques cost and how much would it cut emissions by 2030?',
      'Investing $1.4 billion could cut emissions from rice production by 6.8 MtCO2e relative to BAU by 2030.', [[ev(60, 'text', 'investing $1.4 billion to implement AWD and related techniques could reduce emissions from rice production by 6.8 MtCO2e relative to BAU by 2030')]]),
    q('at24', 'test', 'text', 'Of the credit institutions reporting to the SBV, how many lack a dedicated green finance unit and how many lack a green credit appraisal procedure?',
      'Of 85 credit institutions, 72 lack a dedicated green finance unit and 74 lack a specific green credit appraisal procedure.', [[ev(72, 'text', 'Of 85 credit institutions reporting to the SBV, 72 lack a dedicated business unit for green finance, and 74 lack a specific procedure on green credit appraisal')]]),
    u('ut01', 'test', 'How many dengue fever cases in Vietnam were attributed to climate change in 2020?',
      GREP + "'dengue|malaria' -> 0 pages. Health impacts are discussed only as air pollution deaths, heat mortality (p40) and a $235 million health cost (Figure 3)."),
    u('ut02', 'test', 'How much economic damage did Typhoon Molave cause in Vietnam in 2020?',
      GREP + "'Molave' -> 0 pages; no named typhoon or storm appears ('(Typhoon|Storm) <Name>' -> 0). Storm losses are given only as averages (p28: $2.4 billion per year)."),
    u('ut03', 'test', 'At what price per tonne of CO2 did Vietnam\'s emissions trading system trade in its first year?',
      GREP + "'emission(s) trading|ETS' -> pages 22, 45, 79, 95: the trading system is described as under preparation or recommended; no traded price. Carbon prices given are tax options and the EPT."),
    u('ut04', 'test', 'How much green hydrogen production capacity does Vietnam plan to build by 2030?',
      GREP + "'hydrogen' -> 0 pages."),
    u('ut05', 'test', 'What is the annual mean PM2.5 concentration measured in Ho Chi Minh City?',
      GREP + "'PM2.5' -> pages 19, 47, 79, 90: national averages and Hanoi only; no page mentions both HCMC and PM2.5."),
    u('ut06', 'test', 'What are the annual GHG emissions of the Formosa Ha Tinh steel plant?',
      GREP + "'Formosa|steel plant' -> 0 pages; steel appears only as a sector (e.g. SOE share in Figure 7)."),
]

if __name__ == '__main__':
    ids = [x['id'] for x in QUESTIONS]
    assert len(ids) == len(set(ids))
    out = Path(__file__).with_name('questions.json')
    out.write_text(json.dumps(QUESTIONS, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    from collections import Counter
    print(Counter((x['split'], x['group']) for x in QUESTIONS))
