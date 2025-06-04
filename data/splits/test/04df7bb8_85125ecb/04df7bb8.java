import java.math.BigInteger;
										import java.sql.Array;
									    import java.util.ArrayList;
										import java.util.Arrays;
										import java.util.Collection;
										import java.util.Collections;
										import java.util.Comparator;
										import java.util.Deque;
										import java.util.HashMap;
										import java.util.HashSet;
										import java.util.InputMismatchException;
										import java.util.LinkedList;
										import java.util.PriorityQueue;
										import java.util.Queue;
										import java.util.Scanner;
										import java.util.Set;
										import java.util.Stack;
										import java.io.BufferedReader;
										import java.io.IOException;
										import java.io.InputStream;
										import java.io.InputStreamReader;
										import java.util.Scanner;
										import java.util.StringTokenizer;
										import java.util.TreeMap;
										import java.util.TreeSet;
										
											
											
											
											
											
											public class Main {
							
												
												static HashMap<Integer,Boolean>map;
												
											static long dp[][];
											static 	boolean flag;
										static	HashSet<Long>hs;
											static long mod=(long)(1e9+7);
												public static void main(String[] args) {
												StringBuilder ans=new StringBuilder();
												FastReader sc=new FastReader();
													
												
												int t=sc.nextInt();
												
												
												while(t-->0) {
													
													int n=sc.nextInt();
													
													
													//int n=sb.length();
													
													
													
													int k=sc.nextInt();
													
													
													long L[]=new long[n];
													long R[]=new long[n];
													int a[]=new int[k];
													int temp[]=new int[k];
													
													for(int i=0;i<k;i++)
														a[i]=sc.nextInt();
													
													for(int i=0;i<k;i++)
														temp[i]=sc.nextInt();
													
													int c[]=new int [n];
													Arrays.fill(c, Integer.MAX_VALUE);
													for(int i=0;i<k;i++)
														c[a[i]-1]=temp[i];
												
													long p=Integer.MAX_VALUE;
													
													for(int i=0;i<n;i++)
													{
														p=Math.min(p+1, c[i]);
														
														L[i]=p;
														
														
													}
													p=Integer.MAX_VALUE;
													for(int i=n-1;i>=0;i--)
													{
														p=Math.min(p+1, c[i]);
														
														R[i]=p;
														
														
													}
													
													for(int i=0;i<n;i++)
													{
														ans.append(Math.min(L[i], R[i])+" ");
													}
											ans.append("\n");
													
												}
												System.out.println(ans);
												
												}
													
											
													


												
												
							
												
												
												
												
												
												
												
												
											
												
												
												
												
												
												
												
												
												
												
												
												
												
													
													
												
												
												
												
												
												
												
												
												
												
												
												
												
												
												
									
												
												
											
												
										
												
												
												
												
												
											 
												
											 
												
												
												
											
												
											
												
												
												
												
												
												 
												  
												
												   
												
									
												//System.out.println(count);
												
											
												
												
												
												
												
												
										
										
												
										
												
												
												
												
												//	System.out.println(max);
										
												
											
												
											
									
				//System.out.println(ans.toString());
				//main(						
											
							
					
				
						
												
												
												
								
												
								
												
												
								
												
												
												
												
												
												
												
									
									
												
												
												
												
								
								
								
							
								
									
												
												
												
								
												
												
												
									
							
												
									
									
												
												
												
												
												
												
												
												
												
												
												
									
										
										
										
										
				
				
					
				
				
										
											
																					
											
									    
									 
										
											}
