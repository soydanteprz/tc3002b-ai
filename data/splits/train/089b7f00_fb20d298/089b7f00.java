import java.util.*;
import java.lang.*;
import java.io.*;
public class Main
{
	public static void main (String[] args) throws java.lang.Exception
	{
		Scanner scan=new Scanner(System.in);
		int t=scan.nextInt();
		while(t>0){
		    int n=scan.nextInt();
		    int[] a=new int[n];
		    for(int i=0;i<n;i++){
		        a[i]=scan.nextInt();
		    }
		    TreeSet<Integer>set=new TreeSet<>();
		    set.add(a[0]);
		    boolean flag=false;
		    for(int i=1;i<n;i++){
		        set.add(a[i]);
		        //Collections.sort(list);
		        if(a[i-1]==a[i]) {
					continue;
				}
		        Integer find = set.lower(a[i]);
				if(find!=null && find==a[i-1]) {
					continue;
				}
				find = set.higher(a[i]);
				if(find!=null && find==a[i-1]) {
					continue;
				}
		        //System.out.println(set+" "+x+" "+y);
		      //  if(x==y){
		      //      continue;
		      //  }
		      //  if(x+1==y || x==y+1){
		      //      continue;
		      //  }
		        flag=true;
		        break;
		    }
		    if(flag){
		        System.out.println("No");
		    }else{
		        System.out.println("Yes");
		    }
		    t--;
		}
	}
}
