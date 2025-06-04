import java.io.*;
import java.util.*;

public class GFG {
	public static void main (String[] args) {
	Scanner sc=new Scanner(System.in);
	int t=sc.nextInt();
	while(t-->0){
	    int n=sc.nextInt();
	    String a=sc.next();
	    String b=sc.next();
	   int i;
	   int zero=0,one=0;
	     int x=0,y=0,x1=0,y1=0;
	    for(i=0;i<n;i++){
	        if(a.charAt(i)=='0')
	        zero++;
	        else
	        one++;
	    }
	    if(one==0){
	        if(a.equals(b))
	        System.out.println("0");
	        else
	        System.out.println("-1");
	    }
	    else{
	        int same=0,diff=0;
	        for(i=0;i<n;i++){
	            if(a.charAt(i)==b.charAt(i)){
	                same++;
	                x++;
	                if(a.charAt(i)=='1'){
	                    x1++;
	                    
	                }
	            }
	            else{
	                diff++;
	                y++;
	                if(a.charAt(i)=='1'){
	                    y1++;
	                    
	                }
	            }
	            
	        }
	        int ans=Integer.MAX_VALUE;
    if(x%2!=0&&(x+1)/2==x1){
        ans=x;
        
    }
    if(y%2==0&&(y/2)==y1){
ans=Math.min(ans,y);
    }
    if(ans==Integer.MAX_VALUE){
        
        System.out.println("-1");
        
    }else{
        
        System.out.println(ans);
    }
	       
	    }
	    
	}
	    
	}
    
}
