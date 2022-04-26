window.onload = function() {//0.5秒後に赤くなる
    var changeColor = function() {
        var e = document.getElementById('test');
        e.style.color = 'red';
        console.log("書き換えテスト")
    }
    setTimeout(changeColor, 500); 
}