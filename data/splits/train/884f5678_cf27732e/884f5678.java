import java.io.*;
import java.util.*;
import java.text.*;
import java.math.*;
import java.util.regex.*;

public class JaiShreeRam{
	static Scanner in=new Scanner();
	static long mod = 1000000007;
	static ArrayList<ArrayList<Integer>> adj;
	public static void main(String[] args) throws Exception{
		
		int z=in.readInt();
		while(z-->0) {
			int n=in.readInt();
			int a[]=nia(n);
			char c[]=in.readString().toCharArray();
			ArrayList<Integer> d=new ArrayList<>();
			ArrayList<Integer> in=new ArrayList<>();
			for(int i=0;i<n;i++) {
				if(c[i]=='R') {
					in.add(a[i]);
				}
				else {
					d.add(a[i]);
				}
			}
			String ans="YES";
			Collections.sort(d);
			int k=1;
			for(int i:d) {
				if(i<k) {
					ans="NO";
				}
				k++;
			}
			Collections.sort(in);
			for(int i=in.size()-1;i>=0;i--) {
				if(in.get(i)>n) {
					ans="NO";
					break;
				}
				n--;
			}
			System.out.println(ans);
		}
	}
	
	static int[] nia(int n){
		int[] arr= new int[n];
		int i=0;
		while(i<n){
			arr[i++]=in.readInt();
		}
		return arr;
	}
	
	
	
		
}
