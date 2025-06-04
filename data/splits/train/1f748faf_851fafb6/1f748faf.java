import java.util.*;

import javax.swing.text.html.HTMLDocument.HTMLReader.PreAction;

import java.io.*;
import java.math.*;
import java.sql.Array;;



public class Main {
	  
	 
	  static long mod = (long)(1e9 + 7);
	 
	
	
	
	
	
	
	 
	 
	 
	 
	 
	   
	      

/* ***************************************************************************************************************************************************/	 
	 
	      static FR sc = new FR();
	 static StringBuilder sb = new StringBuilder();
	 public static void main(String args[]) {
		
		 int tc = 1;
		  tc = sc.nextInt();
		 
		 
		 while(tc-->0) {
			 TEST_CASE();
			 
		 }
		 
	    System.out.println(sb);
	 }


	 static void TEST_CASE() {
		 int n = sc.nextInt(); ;
		 int[] arr = new int[n];
		 for(int i =0 ; i<n;i++) {
			 arr[i] = sc.nextInt();
		 } 
		 boolean cond = false;
		
		 Set<Integer> set = new HashSet<>();
		 set.add(0);
		 for(int i =0 ; i<n ; i++) {
			 ArrayList<Integer> al = new ArrayList<>(set);
			 for(int e:al) {
				 int num = e+arr[i];
				 if(set.contains(num)) {
					 cond = true;
					 break;
				 }
				 set.add(num);
			 }
		 }
		 if(cond) System.out.println("YES");
		 else System.out.println("NO");
	 }

	 
}
