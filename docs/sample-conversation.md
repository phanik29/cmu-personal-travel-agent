# Sample conversation

A full chat with the app, recorded in Chrome on November 22, 2026, with Duffel test flights, mock hotels, cars, and pickups, live weather, and the local model `gemma4:12b`. Button presses appear as the text the chat shows for them, and every booking is a sandbox booking.

**You:** Plan a trip from San Francisco to New York, Dec 22 to Dec 27, 2 people, budget $3000, balanced, and we'd like to land at night

**Agent:**

> Here's my plan for your trip: San Francisco (SFO) to New York (NYC), Tue, Dec 22 to Sun, Dec 27, 2 travelers, budget $3,000, landing at night, optimizing for a balance of price and convenience.
>
> Airports:
> - From: San Francisco International Airport (SFO)
> - To: New York, any airport (JFK, LGA, or EWR)
>
> What I'll do:
> 1. Check the weather and airports for New York (NYC)
> 2. Flight agent: search flights from SFO to NYC (best overall, nonstop, cheapest), keeping only flights that land at night (9 PM to 5 AM)
> 3. Hotel agent: search hotels for 5 nights (best rated)
> 4. Car rental agent: search rental cars (standard)
> 5. Transfer agent: search airport pickups for when you land (standard)
> 6. Combine flights, hotels, cars, and airport pickups and keep the best 3 at each step (Tree-of-Thought beam search)
> 7. Check budget, dates, availability, and arrival time against hotel check-in and rental desk hours, then rank by a balance of price and convenience
> 8. If something doesn't fit, run a follow-up search or ask you
> 9. Show you the best options; nothing is booked until you approve
>
> Next step: say "go ahead" (or press Start search), and the agents will search at the same time. To change anything, just tell me.

**You:** what's the weather like?

**Agent:**

> Weather in New York, Tue, Dec 22 to Sun, Dec 27:
>
> Highs 2 to 8°C, lows -7 to 1°C. Rain likely on 3 of 6 days. Wind up to 29 km/h. Based on the same dates in 2025, not a forecast. Overall risk: low.
>
> Next step: say "go ahead" (or press Start search), and the agents will search at the same time. To change anything, just tell me.

**You:** Go ahead and search

**Agent:**

> I found 3 options for your trip: San Francisco (SFO) to New York (NYC), Tue, Dec 22 to Sun, Dec 27, 2 travelers, budget $3,000, landing at night, optimizing for a balance of price and convenience.
>
> My top pick costs $1,678.08 in total, $1,321.92 under your budget:
> - Flight: British Airways BA0109, nonstop, leaves Tue, Dec 22 at 4:15 PM and arrives Wed, Dec 23 at 1:21 AM; returns Sun, Dec 27 at 8:07 PM ($633.52)
> - Hotel: Old Town Lodge, old town area, rated 4.0, 5 nights, free cancellation until Dec 15 ($670.53)
> - Car: Hertz Mock midsize, desk open 24 hours ($289.65)
> - Airport pickup: City Car Mock private sedan from LGA to Old Town Lodge when you land, about 45 minutes ($84.38)
> - Airports: San Francisco International Airport (SFO) to LaGuardia Airport (LGA)
>
> The top 2 options are very close, so pick the one you like best. All of them are on the right.
>
> Weather in New York: no problems expected (based on past years, not a forecast).
>
> This plan fits you because it is $1,322 under your $3,000 budget. It also offers a nonstop flight and a 24-hour rental desk for a balanced travel experience.
>
> Next step: choose an option on the right to book it in the sandbox (no real payment), or tell me what to change.

**You:** a different hotel

**Agent:**

> Here's my plan to find a different hotel for your trip: San Francisco (SFO) to New York (NYC), Tue, Dec 22 to Sun, Dec 27, 2 travelers, budget $3,000, landing at night, optimizing for a balance of price and convenience. I'll keep the flight, rental car, and airport pickup from my top pick and skip the 1 hotel I already showed you.
>
> Airports:
> - From: San Francisco International Airport (SFO)
> - To: New York, any airport (JFK, LGA, or EWR)
>
> What I'll do:
> 1. Check the weather and airports for New York (NYC)
> 2. Hotel agent: search hotels for 5 nights (best rated, lowest price)
> 3. Combine the hotels I find with the flight, rental car, and airport pickup I'm keeping, and keep the best 3 (Tree-of-Thought beam search)
> 4. Check budget, dates, availability, and arrival time against hotel check-in and rental desk hours, then rank by a balance of price and convenience
> 5. If something doesn't fit, run a follow-up search or ask you
> 6. Show you the best options; nothing is booked until you approve
>
> Next step: say "go ahead" (or press Start search), and the agents will search at the same time. To change anything, just tell me.

**You:** Go ahead and search

**Agent:**

> I found 3 options with a different hotel for your trip: San Francisco (SFO) to New York (NYC), Tue, Dec 22 to Sun, Dec 27, 2 travelers, budget $3,000, landing at night, optimizing for a balance of price and convenience.
>
> I kept the same flight, rental car, and airport pickup.
>
> My top pick costs $1,850.08 in total, $1,149.92 under your budget:
> - Flight: British Airways BA0109, nonstop, leaves Tue, Dec 22 at 4:15 PM and arrives Wed, Dec 23 at 1:21 AM; returns Sun, Dec 27 at 8:07 PM ($633.52)
> - Hotel: Harbor View Suites, waterfront area, rated 4.3, 5 nights, free cancellation until Dec 19 ($842.53)
> - Car: Hertz Mock midsize, desk open 24 hours ($289.65)
> - Airport pickup: City Car Mock private sedan from LGA to Harbor View Suites when you land, about 45 minutes ($84.38)
> - Airports: San Francisco International Airport (SFO) to LaGuardia Airport (LGA)
>
> You can compare it with 2 other options on the right.
>
> Weather in New York: no problems expected (based on past years, not a forecast).
>
> This plan fits you because it is $1,150 under your $3,000 budget. It also offers a nonstop flight and a waterfront hotel with free cancellation until Dec 19.
>
> Next step: choose an option on the right to book it in the sandbox (no real payment), or tell me what to change.

**You:** Book option 1 ($1,850.08)

**Agent:**

> All set. Your trip is booked in the sandbox (no real payment):
> - Flight: British Airways BA0109, SFO to LGA, confirmation SBX-FL-5E612A ($633.52)
> - Hotel: Harbor View Suites, confirmation SBX-HT-3B8C3F ($842.53)
> - Car: Hertz Mock midsize, confirmation SBX-CR-FB715B ($289.65)
> - Airport pickup: City Car Mock private sedan, confirmation SBX-TR-0BDCEC ($84.38)
>
> I checked each booking with the provider, and everything matches what you approved.
>
> Next steps: ask me anything about your trip, cancel a booking (for example, "cancel my car"), or start a new trip.

**You:** cancel my flight

**Agent:**

> Do you want me to cancel this booking?
> - Flight: British Airways BA0109, SFO to LGA, confirmation SBX-FL-5E612A, $633.52 (refundable fare)
>
> Next step: confirm on the right. Nothing is cancelled until you do.

**You:** Yes, cancel it

**Agent:**

> Cancelled:
> - Flight: British Airways BA0109, SFO to LGA, refund $633.52
>
> I checked with the provider, and the cancellation went through.
>
> Next steps: ask me anything about your trip, cancel a booking (for example, "cancel my car"), or start a new trip.
