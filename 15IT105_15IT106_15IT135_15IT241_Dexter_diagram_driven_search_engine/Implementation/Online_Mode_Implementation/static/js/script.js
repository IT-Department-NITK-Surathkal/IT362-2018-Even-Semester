$(document).ready(function () {
    $("#sidenav").sideNav();

    $(".dropdown-button").dropdown();
    $('.dropdown-button').dropdown({
        hover: true, // Activate on hover
        belowOrigin: true, // Displays dropdown below the button
        alignment: 'left', // Displays dropdown with edge aligned to the left of button
    });
    loadInfo()
});

function loadInfo() {
    console.log(results[0]['id'])
    
    for (var i = 0; i < results.length; i++) {
        if (i % 2 === 0) {
            var row = document.createElement('div');
            row.className = "row";
            row.id = "content-row-" + i / 2;

            $('#content-team-container').append(row);
        }
        member = results[i];

        var container = document.createElement('div');
        container.className = "member-container col s12 m6";

        var card = document.createElement('div')
        card.className = "card";

        container.appendChild(card);

        var card_content = document.createElement('div')
        card_content.className = "card-content";

        var title = document.createElement('span');
        title.className = "card-title activator grey-text text-darken-4";
        title.innerText = member.title;

        var activator = document.createElement('i');
        activator.className = "material-icons right";
        activator.innerText = "more_vert";

        card_content.appendChild(title);

        var x_label = document.createElement('p');
        x_label.innerText = 'X-Label:' + member.x_label;
        card_content.appendChild(x_label);

        var y_label = document.createElement('p');
        y_label.innerText = 'Y-Label:' + member.y_label;
        card_content.appendChild(y_label);

        var x_min = document.createElement('p');
        x_min.innerText = 'X-Min:' + member.x_min;
        card_content.appendChild(x_min);

        var y_min = document.createElement('p');
        y_min.innerText = 'Y-Min:' + member.y_min;
        card_content.appendChild(y_min);

        var x_max = document.createElement('p');
        x_max.innerText = 'X-Max:' + member.x_max;
        card_content.appendChild(x_max);

        var y_max = document.createElement('p');
        y_max.innerText = 'Y-Max:' + member.y_max;
        card_content.appendChild(y_max);

        var link = document.createElement('a');
        link.innerText = 'Link to Relevant Document';
        console.log(String(member.file_location));
        link.setAttribute('href', member.file_location);
        card_content.appendChild(link);

        // file - location
        card.appendChild(card_content);

        var card_reveal = document.createElement('div')
        card_reveal.className = "card-reveal";

        var reveal_title = document.createElement('span');
        reveal_title.className = "card-title grey-text text-darken-4";
        reveal_title.innerText = member.title;

        var reveal_icon = document.createElement('i');
        reveal_icon.className = "material-icons right";
        reveal_icon.innerText = 'close';

        reveal_title.appendChild(reveal_icon);

        card_reveal.appendChild(reveal_title);

        var reveal_description = document.createElement('p');
        reveal_description.className = "card-title activator grey-text text-darken-4";
        reveal_description.innerText = member.description;

        card_reveal.appendChild(reveal_description);

        card.appendChild(card_reveal);

        console.log(container.innerHTML);

        $('#content-row-' + parseInt(i / 2)).append(container);
    }
}