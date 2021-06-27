document.querySelector('.img-btn').addEventListener('click', function()
	{
		document.querySelector('.cont').classList.toggle('s-signup')
	}
);


function W(){
	var u = document.getElementById("ur").value;
	var d = document.getElementById("dt").value;
	var h = document.getElementById("hl").value;
	var b = document.getElementById("bd").value;
	var s = document.getElementById("sr").value;
	var t = document.getElementById("dt2").value;
	if((u === "") && (h === "")){
		alert("Please fill the URL/headline field");
	}
	if(u !== ""){
		document.getElementById("urt").innerHTML = u;
		document.getElementById("dtt").innerHTML = d;
	}
	if(h !== ""){
		document.getElementById("hlt").innerHTML = h;
		document.getElementById("bdt").innerHTML = b;
		document.getElementById("srt").innerHTML = s;
		document.getElementById("dt2t").innerHTML = t;
	}

}
