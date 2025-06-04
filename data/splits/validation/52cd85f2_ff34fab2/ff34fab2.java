import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Stack;
import java.util.StringTokenizer;
import java.util.TreeMap;
import java.util.TreeSet;

import java.io.*;
 
public class Div2 {
	
	

	 public static String solution(int [] arr, int n, String st) 
	 {
        ArrayList<Integer> red = new ArrayList<>();
        ArrayList<Integer> blue = new ArrayList<>();
        
        for(int i = 0; i<n; i++)
        {
        	if(st.charAt(i)=='R')
        		red.add(arr[i]);
        	else
        		blue.add(arr[i]);
        }
	    	
        Collections.sort(red);
        Collections.sort(blue);
        int cb = 1;
        
        for(int j = 0; j<blue.size(); j++)
        {
            if(blue.get(j)<cb)
            	return "NO";
            cb++;
        }
        
        int cr = n;
        for(int j = red.size()-1; j>=0; j--)
        {
            if(red.get(j)>cr)
            	return "NO";
            cr--;
        }
        
        return "YES";

	 }
        
private static PrintWriter out = new PrintWriter(System.out);

public static void main (String[] args)
{
	MyScanner s =  new MyScanner();
	  
    int t = s.nextInt();
    
    for(int j = 0; j<t ; j++)
    {
    	int n = s.nextInt();
    	int[] arr = new int[n];
    	for(int i =0; i<n; i++)
    		arr[i] = s.nextInt();
    	
    	String st = s.next();
   
    	out.println(solution(arr,n, st));
    }
    
		     
    out.flush();
    out.close();
    
}
}
