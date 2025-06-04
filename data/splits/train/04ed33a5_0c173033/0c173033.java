import java.io.*;
import java.util.*;


public class M {
	static Scanner scanner=new Scanner(System.in);
	public static void main(String[] args) {
		int t=scanner.nextInt();
		while(t-->0) {
			int n=scanner.nextInt();
			int a[]=new int [n];
			int b[]=new int [n];
			for(int i=0;i<n;i++)a[i]=scanner.nextInt();
			for(int i=0;i+1<n;i+=2) {
				b[i]=-a[i+1];
				b[i+1]=a[i];
			}
			if(n%2==1) {
				int x=a[n-1],y=a[n-2],z=a[n-3];
				if(x+y!=0) {
					b[n-3]=x+y;
					b[n-2]=-z;
					b[n-1]=-z;
				}else if(y+z!=0) {
					b[n-1]=y+z;
					b[n-2]=-x;
					b[n-3]=-x;
				}else {
					b[n-2]=x+z;
					b[n-1]=-y;
					b[n-3]=-y;
				}
			}
			StringBuilder sb=new StringBuilder();
			for(int i=0;i<n;i++)sb.append(b[i]+" ");
			System.out.println(sb);
		}
	}
}
