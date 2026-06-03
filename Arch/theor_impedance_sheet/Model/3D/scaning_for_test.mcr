' scaning_for_test

Sub ExportArrayToCSV(arr() As Double, fileName As String, mode As Long)
	Dim filePath As String, i As Long, j As Long
	filePath = GetProjectPath("Project") & "\" & fileName & ".csv"
	Open filePath For Output As #1
	If mode = 1 Then
		Debug.Print "mode ="; 1
		Debug.Print filePath
		For i = LBound(arr) To UBound(arr)
			Dim aline As String
			aline = Cstr(arr(i))
			Print #1, aline
		Next
	End If
	If mode = 2 Then
		Debug.Print "mode ="; 2
		Debug.Print filePath
		For i = LBound(arr, 1) To UBound(arr, 1)
			Dim aline2 As String
			aline2 = ""
			For j = LBound(arr, 2) To UBound(arr, 2)
				aline2 = aline2 & Cstr(arr(i,j))
				If j<UBound(arr, 2) Then aline2 = aline2 & ","
			Next
			Print #1, aline2
		Next
	End If
	Close #1
End Sub

Function Linspace(startVal As Double, endVal As Double, n As Long) As Variant
	Dim arr() As Double
    ReDim arr(0 To n - 1)
    Dim i As Long
    Dim stepVal As Double
    stepVal = (endVal - startVal) / n

    For i = 0 To n-1
        arr(i) = startVal + i * stepVal
    Next i

    Linspace = arr
End Function

Sub Main ()
	Debug.Clear
	Dim n As Long, i As Long
	n = 50
	Dim res() As Double
	ReDim res(0 To n-1, 0 To 1)
	Dim theta() As Double
	ReDim theta(0 To n-1)
	theta = Linspace(0, 90, n)
	For i = 0 To n-1
		Dim id As Variant, path As String, myProbe_horiz As Object, tupe As String
		StoreParameter("scan_theta", Cstr(theta(i)))
		DeleteResults
		RunSolver()
		With CombineResults
		    .Reset
		    .SetMonitorType ("frequency")
		    .EnableAutomaticLabeling (False)
		    .SetLabel ("horiz")
		    .SetNone
		    .SetExcitationValues ("port", "Zmax", 1, "scale_for_amplitude", 0)
		    .Run
		End With
		path = "1D Results\Probes\E-Field\E-Field (0 0 59.9585)(X) [horiz]"
		With Resulttree
			id = .GetResultIDsFromTreeItem(path)
			Set myProbe_horiz = .GetResultFromTreeItem(path, Cstr(id(1)))
		End With
		With CombineResults
		    .Reset
		    .SetMonitorType ("frequency")
		    .EnableAutomaticLabeling (False)
		    .SetLabel ("vert")
		    .SetNone
		    .SetExcitationValues ("port", "Zmax", 2, "scale_for_amplitude", 0)
		    .Run
		End With
		path = "1D Results\Probes\E-Field\E-Field (0 0 59.9585)(X) [vert]"
		With Resulttree
			id = .GetResultIDsFromTreeItem(path)
			Set myProbe_vert = .GetResultFromTreeItem(path, Cstr(id(1)))
		End With
		res(i,0) = theta(i)
		res(i,1) = Sqr(myProbe_horiz.GetYRe(0)^2 + myProbe_horiz.GetYIm(0)^2+myProbe_vert.GetYRe(0)^2 + myProbe_vert.GetYIm(0)^2)
	Next
	ExportArrayToCSV(res, "H-plane", 2)
	MsgBox "Âñ¸ Îê"
End Sub
